# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
=============================================================================
  SurfaceSync - Transmitter Backend  (transmitter_backend.py)
  Run : python transmitter_backend.py
  Port: http://0.0.0.0:5001

  Pipeline
  --------
  1. File/message upload from transmitter.html
  2. Split into SUBPART_SIZE (8 KB) sub-parts
  3. Encrypt each sub-part with AES-256-CBC (independent IV per sub-part)
  4. Split each encrypted sub-part into PACKET_SIZE (256 B) chunks
  5. Build wire packets (sync + header + [IV] + ciphertext chunk + CRC)
  6. Wrap each packet in CMD_PACKET serial command -> ESP32-CAM
  7. ESP32-CAM plays 4-FSK audio through exciter

  Surface sweep
  -------------
  1. Send CMD_SWEEP -> ESP32-CAM plays 1000-6500 Hz tones
  2. Receiver Python backend POSTs sweep FFT result to /surface_feedback
  3. Transmitter backend computes 4-FSK frequencies -> sends CMD_SET_FREQ

  4-FSK frequency table (750 Hz spacing)
  ---------------------------------------
  Surface    base    00      01      10      11    preamble
  Glass      2000  2000   2750   3500   4250    5000
  Hardwood   1500  1500   2250   3000   3750    4500
  Softwood   1000  1000   1750   2500   3250    4000
  Metal      3500  3500   4250   5000   5750    6500

  Dependencies
  ------------
  pip install flask flask-socketio flask-cors pycryptodome pyserial
=============================================================================
"""

import os
import time
import struct
import threading
import queue

import serial
import serial.tools.list_ports

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Hash import SHA256

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(32)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Acoustic / protocol constants ─────────────────────────────────────────────
SUBPART_SIZE  = 8192    # bytes per sub-part (plaintext)
PACKET_SIZE   = 256     # max ciphertext bytes per packet
RETRY_LIMIT   = 5
CMD_TIMEOUT   = 3.0     # seconds to wait for ACK

# Serial command bytes (must match ESP32 firmware)
SYNC          = bytes([0xAA, 0x55])
ACK_BYTE      = 0x06
NACK_BYTE     = 0x15
SWEEP_DONE    = 0xA0
CMD_SWEEP     = 0x01
CMD_SET_FREQ  = 0x02
CMD_PACKET    = 0x03
CMD_RESET     = 0x04
CMD_GUARD     = 0x05

# 4-FSK base frequencies per surface (Hz)
SURFACE_BASE = {
    "glass"    : 2000,
    "hardwood" : 1500,
    "softwood" : 1000,
    "metal"    : 3500,
    "unknown"  : 2000,   # default to glass
}
FREQ_SPACING = 750

# ── Runtime state ─────────────────────────────────────────────────────────────
STATE = {
    "transmitting" : False,
    "serial_port"  : None,
    "surface"      : "unknown",
    "base_freq"    : 2000,
    "freqs"        : [2000, 2750, 3500, 4250, 5000],
    "aes_password" : "SurfaceSyncSecretKey2025",
}

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    socketio.emit("log", {"message": msg, "level": level})
    print(f"[TX-{level}] {msg}")

# ── CRC-16/IBM (must match ESP32 firmware) ────────────────────────────────────
def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else crc >> 1
    return crc & 0xFFFF

# ── AES-256-CBC helpers ───────────────────────────────────────────────────────
def derive_key(password: str) -> bytes:
    return SHA256.new(password.encode()).digest()

def encrypt_subpart(plaintext: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Return (iv, ciphertext) for one sub-part."""
    cipher     = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return cipher.iv, ciphertext

# ── Frequency helpers ─────────────────────────────────────────────────────────
def compute_freqs(base_freq: int) -> list[int]:
    """Return [f00, f01, f10, f11, f_preamble] for given base frequency."""
    return [base_freq + i * FREQ_SPACING for i in range(5)]

# ── Serial command framing ────────────────────────────────────────────────────
def build_serial_frame(cmd: int, data: bytes = b"") -> bytes:
    """
    Build: [0xAA][0x55][CMD][LEN_HI][LEN_LO][DATA][CRC_HI][CRC_LO]
    CRC-16 over DATA bytes only.
    """
    length = len(data)
    crc    = crc16(data) if data else 0xFFFF
    frame  = bytearray()
    frame += SYNC
    frame.append(cmd)
    frame.append((length >> 8) & 0xFF)
    frame.append(length & 0xFF)
    frame += data
    frame.append((crc >> 8) & 0xFF)
    frame.append(crc & 0xFF)
    return bytes(frame)

def send_serial_cmd(cmd: int, data: bytes, ser: serial.Serial,
                    retries: int = RETRY_LIMIT) -> bool:
    """
    Send one command frame to ESP32-CAM; wait for ACK.
    Returns True on ACK, False after all retries exhausted.
    """
    frame = build_serial_frame(cmd, data)
    for attempt in range(retries):
        ser.write(frame)
        deadline = time.time() + CMD_TIMEOUT
        while time.time() < deadline:
            if ser.in_waiting:
                resp = ser.read(1)
                if resp and resp[0] == ACK_BYTE:
                    return True
                if resp and resp[0] == NACK_BYTE:
                    break   # NACK -> retry
            time.sleep(0.001)
        log(f"CMD 0x{cmd:02X} attempt {attempt+1} failed - retrying", "ERROR")
    return False

# ── Packet builder ────────────────────────────────────────────────────────────
def build_acoustic_packet(sub_id:    int,
                           sub_total: int,
                           pkt_id:    int,
                           pkt_total: int,
                           iv:        bytes | None,
                           chunk:     bytes) -> bytes:
    """
    Build the packet bytes that will be transmitted as 4-FSK audio.

    Wire format:
        [0xAA][0x55]        2B  acoustic sync word
        [sub_id]            2B  which sub-part
        [sub_total]         2B  total sub-parts
        [pkt_id]            2B  packet index within sub-part
        [pkt_total]         2B  total packets in this sub-part
        [flags]             1B  bit0=1 means IV follows
        [data_len]          2B  bytes of payload following
        [iv: 16B]           optional, only when flags & 0x01
        [ciphertext chunk]  data_len bytes
        [pkt_crc]           2B  CRC-16 of everything after acoustic sync

    Returns the full byte sequence to be handed to CMD_PACKET.
    """
    flags    = 0x01 if iv is not None else 0x00
    payload  = (iv or b"") + chunk
    data_len = len(payload)

    header = struct.pack(">HHHHHBH",
                         sub_id, sub_total,
                         pkt_id, pkt_total,
                         flags, data_len)   # wait - BH mismatch
    # Let's build manually to be explicit:
    header = (
        struct.pack(">H", sub_id)    +
        struct.pack(">H", sub_total) +
        struct.pack(">H", pkt_id)    +
        struct.pack(">H", pkt_total) +
        struct.pack("B",  flags)     +
        struct.pack(">H", data_len)
    )

    body     = header + payload
    pkt_crc  = crc16(body)
    packet   = b"\xAA\x55" + body + struct.pack(">H", pkt_crc)
    return packet

# ── Sub-part processing ───────────────────────────────────────────────────────
def build_all_packets(raw_data: bytes, password: str) -> list[bytes]:
    """
    Split raw_data into sub-parts -> encrypt -> build all acoustic packets.
    Returns a flat list of packet byte strings ready for CMD_PACKET.
    """
    key        = derive_key(password)
    subparts   = [raw_data[i : i + SUBPART_SIZE]
                  for i in range(0, len(raw_data), SUBPART_SIZE)]
    sub_total  = len(subparts)
    all_packets: list[bytes] = []

    log(f"Building packets: {len(raw_data):,} B / {sub_total} sub-parts / "
        f"{SUBPART_SIZE} B each", "INFO")

    for sub_id, subpart in enumerate(subparts):
        iv, ciphertext = encrypt_subpart(subpart, key)

        # Split ciphertext into 256-byte chunks
        chunks    = [ciphertext[i : i + PACKET_SIZE]
                     for i in range(0, len(ciphertext), PACKET_SIZE)]
        pkt_total = len(chunks)

        for pkt_id, chunk in enumerate(chunks):
            iv_to_send = iv if pkt_id == 0 else None
            pkt = build_acoustic_packet(sub_id, sub_total,
                                         pkt_id, pkt_total,
                                         iv_to_send, chunk)
            all_packets.append(pkt)

    log(f"[OK] {len(all_packets)} acoustic packets built "
        f"({sub_total} sub-parts x ~{len(subparts[0])//PACKET_SIZE+1} pkts)", "DATA")
    return all_packets

# ── Surface sweep ─────────────────────────────────────────────────────────────
def run_surface_sweep(ser: serial.Serial) -> tuple[str, int]:
    """
    Send CMD_SWEEP to ESP32-CAM; block until SWEEP_DONE (0xA0) received.
    Then wait up to 30 s for the receiver backend to POST /surface_feedback.
    Returns (surface_type, base_freq).
    """
    log("Starting surface sweep (1000-6500 Hz) ...", "INFO")
    frame = build_serial_frame(CMD_SWEEP, b"")
    ser.write(frame)

    # Wait for ACK (ESP32 acks before starting sweep)
    deadline = time.time() + CMD_TIMEOUT
    while time.time() < deadline:
        if ser.in_waiting and ser.read(1)[0] == ACK_BYTE:
            break
        time.sleep(0.001)

    # Wait for SWEEP_DONE
    deadline = time.time() + 30
    while time.time() < deadline:
        if ser.in_waiting and ser.read(1)[0] == SWEEP_DONE:
            log("Sweep tones finished. Waiting for RX feedback ...", "INFO")
            break
        time.sleep(0.01)

    # Wait for receiver backend to post the result
    deadline = time.time() + 30
    while time.time() < deadline:
        if STATE.get("_sweep_result"):
            surface, base_freq = STATE.pop("_sweep_result")
            return surface, base_freq
        time.sleep(0.2)

    log("Sweep timeout - using default Glass frequencies.", "ERROR")
    return "unknown", SURFACE_BASE["unknown"]

def apply_surface_frequencies(surface: str, base_freq: int,
                               ser: serial.Serial | None) -> None:
    """
    Compute 4-FSK frequencies from surface base freq and notify ESP32-CAM.
    """
    freqs = compute_freqs(base_freq)
    STATE["surface"]   = surface
    STATE["base_freq"] = base_freq
    STATE["freqs"]     = freqs

    log(f"Surface: {surface} | Freqs: {freqs[0]}/{freqs[1]}/{freqs[2]}/{freqs[3]} Hz "
        f"(preamble {freqs[4]} Hz)", "SUCCESS")
    socketio.emit("sweep_result", {"surface": surface, "freqs": freqs})

    if ser and ser.is_open:
        # CMD_SET_FREQ: 5 x uint16 big-endian = 10 bytes
        freq_data = b"".join(struct.pack(">H", f) for f in freqs)
        ok = send_serial_cmd(CMD_SET_FREQ, freq_data, ser)
        if ok:
            log("Frequencies pushed to ESP32-CAM.", "SUCCESS")
        else:
            log("Failed to push frequencies to ESP32-CAM.", "ERROR")

# ── Main transmission pipeline ────────────────────────────────────────────────
def run_transmission_pipeline(raw_data: bytes, password: str) -> None:
    """Full TX pipeline. Runs in a background thread."""
    STATE["transmitting"] = True
    socketio.emit("status", {"state": "transmitting"})

    try:
        ser = STATE.get("serial_port")
        dry_run = (ser is None or not ser.is_open)

        if dry_run:
            log("No serial port - dry-run mode (no hardware).", "INFO")

        # 1. Surface sweep
        if not dry_run:
            surface, base_freq = run_surface_sweep(ser)
            apply_surface_frequencies(surface, base_freq, ser)
        else:
            log("Dry-run: skipping sweep, using Glass defaults.", "INFO")

        # 2. Build all packets
        packets   = build_all_packets(raw_data, password)
        total     = len(packets)

        # 3. Transmit
        sent = 0
        for pkt_bytes in packets:
            if dry_run:
                # Simulate progress for UI testing
                time.sleep(0.005)
                ok = True
            else:
                ok = send_serial_cmd(CMD_PACKET, pkt_bytes, ser)

            if ok:
                sent += 1
            else:
                log(f"Packet {sent} failed after {RETRY_LIMIT} retries.", "ERROR")

            progress = (sent / total) * 100.0
            socketio.emit("progress", {
                "percent" : round(progress, 1),
                "sent"    : sent,
                "total"   : total,
            })
            if sent % max(1, total // 20) == 0:
                log(f"Packet {sent}/{total} ({progress:.0f}%)", "DATA")

        if sent == total:
            log(f"[OK] Transmission complete - {sent} packets sent.", "SUCCESS")
            socketio.emit("status", {"state": "done"})
        else:
            log(f"[FAIL] {total - sent} packets lost.", "ERROR")
            socketio.emit("status", {"state": "error"})

    except Exception as exc:
        log(f"Pipeline exception: {exc}", "ERROR")
        socketio.emit("status", {"state": "error"})
    finally:
        STATE["transmitting"] = False

# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return "<h2>SurfaceSync Transmitter Backend - :5001</h2>"


@app.route("/upload", methods=["POST"])
def upload():
    """
    Accept file (multipart) or JSON {message, password}.
    Starts transmission pipeline in background thread.
    """
    if STATE["transmitting"]:
        return jsonify({"error": "Transmission already in progress"}), 409

    password = STATE["aes_password"]
    raw_data = None

    if request.files.get("file"):
        f        = request.files["file"]
        raw_data = f.read()
        password = request.form.get("password", password)
        log(f"File received: '{f.filename}' ({len(raw_data):,} B)", "DATA")
    elif request.is_json:
        body     = request.get_json()
        raw_data = body.get("message", "").encode("utf-8")
        password = body.get("password", password)
        log(f"Text message received ({len(raw_data)} B)", "DATA")
    else:
        return jsonify({"error": "Send multipart file or JSON {message}"}), 400

    if not raw_data:
        return jsonify({"error": "Empty payload"}), 400

    t = threading.Thread(
        target=run_transmission_pipeline,
        args=(raw_data, password),
        daemon=True,
    )
    t.start()
    return jsonify({"status": "queued", "size_bytes": len(raw_data)}), 202


@app.route("/surface_feedback", methods=["POST"])
def surface_feedback():
    """
    Called by the Receiver backend with FFT sweep result.
    JSON: { "surface": "glass", "base_freq": 2000 }
    """
    body     = request.get_json(force=True)
    surface  = body.get("surface", "unknown").lower()
    freq     = int(body.get("base_freq", SURFACE_BASE.get(surface, 2000)))
    STATE["_sweep_result"] = (surface, freq)
    log(f"Sweep feedback received: {surface} @ {freq} Hz", "SUCCESS")
    return jsonify({"ack": True}), 200


@app.route("/connect_serial", methods=["POST"])
def connect_serial():
    """JSON: { "port": "/dev/ttyUSB0" }  (port optional - auto-detect)"""
    body = request.get_json(force=True) or {}
    port = body.get("port")
    baud = int(body.get("baud", 921600))

    if not port:
        for p in serial.tools.list_ports.comports():
            if any(k in p.description.lower() for k in ["cp210", "ch340", "uart", "usb serial"]):
                port = p.device
                break

    if not port:
        return jsonify({"error": "No serial port found"}), 404

    try:
        ser = serial.Serial(port, baud, timeout=2.0)
        STATE["serial_port"] = ser
        log(f"Serial connected: {port} @ {baud}", "SUCCESS")
        return jsonify({"status": "connected", "port": port}), 200
    except serial.SerialException as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set_key", methods=["POST"])
def set_key():
    body = request.get_json(force=True)
    pwd  = body.get("password", "")
    if len(pwd) < 8:
        return jsonify({"error": "Password must be >= 8 chars"}), 400
    STATE["aes_password"] = pwd
    log("AES password updated.", "INFO")
    return jsonify({"status": "ok"}), 200


@app.route("/list_ports", methods=["GET"])
def list_ports():
    ports = [{"device": p.device, "description": p.description}
             for p in serial.tools.list_ports.comports()]
    return jsonify({"ports": ports}), 200


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "transmitting" : STATE["transmitting"],
        "surface"      : STATE["surface"],
        "base_freq"    : STATE["base_freq"],
        "freqs"        : STATE["freqs"],
        "serial_open"  : bool(STATE.get("serial_port") and STATE["serial_port"].is_open),
    }), 200


# ── Socket.IO ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("status", {
        "state"   : "transmitting" if STATE["transmitting"] else "idle",
        "surface" : STATE["surface"],
        "freqs"   : STATE["freqs"],
    })


@socketio.on("ping_backend")
def on_ping():
    emit("pong", {"ts": time.time()})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SurfaceSync - Transmitter Backend")
    print("  http://0.0.0.0:5001")
    print("  POST /upload           -> start transmission")
    print("  POST /surface_feedback -> RX sweep result")
    print("  POST /connect_serial   -> open ESP32-CAM port")
    print("  GET  /list_ports       -> COM port list")
    print("  GET  /status           -> pipeline state")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)