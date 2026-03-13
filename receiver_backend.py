# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

"""
=============================================================================
  SurfaceSync - Receiver Backend  (receiver_backend.py)
  Run : python receiver_backend.py
  Port: http://0.0.0.0:5002

  Pipeline
  --------
  1. Ingest raw ADC int16 frames from ESP32-WROOM via serial
  2. Bandpass filter (scipy) -> pre-filter before CNN
  3. 1D-CNN binary classifier: signal window vs noise window
  4. 4-FSK demodulator: zero-padded FFT energy at 4 symbol frequencies
  5. Byte stream parser: hunt for 0xAA 0x55 sync -> parse packet header
  6. Sub-part store: collect all packets per sub-part ID
  7. AES-256-CBC decrypt each complete sub-part (IV from first packet)
  8. Reassemble all sub-parts -> original file
  9. Surface sweep: FFT of ESP32-WROOM sweep audio -> POST to TX backend
 10. Real-time Socket.IO events to receiver.html

  Serial frame from ESP32-WROOM
  --------------------------------
  [0xBB][0x44][LEN_HI][LEN_LO][int16 samples BE][CRC_HI][CRC_LO]
  LEN_HI bit7=1 -> preamble detected by ESP32 hardware gate
  Actual data length = LEN_WORD & 0x7FFF

  Dependencies
  ------------
  pip install flask flask-socketio flask-cors pycryptodome numpy \
              pyserial scipy tensorflow requests
=============================================================================
"""

import os
import time
import struct
import threading
import queue
import json

import numpy as np
import serial
import serial.tools.list_ports
import requests

from scipy.signal import butter, sosfilt

from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from Crypto.Hash import SHA256

# ── TensorFlow (optional) ─────────────────────────────────────────────────────
try:
    import tensorflow as tf              # type: ignore[import-untyped]
    import tensorflow.keras as keras     # type: ignore[import-untyped]
    TF_AVAILABLE = True
    print("[RX] TensorFlow loaded - CNN mode active.")
except ImportError:
    tf    = None                         # type: ignore[assignment]
    keras = None                         # type: ignore[assignment]
    TF_AVAILABLE = False
    print("[RX] TensorFlow not found - bandpass-only filter.")

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(32)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
BAUD_SYMBOLS  = 1000
SPS           = SAMPLE_RATE // BAUD_SYMBOLS     # samples per symbol = 44
FFT_PAD       = 4096                             # zero-pad for fine resolution
FREQ_SPACING  = 750                              # Hz between FSK symbols
CNN_WINDOW    = 512                              # samples per CNN inference window
CNN_STRIDE    = 128
CNN_THRESHOLD = 0.60

SUBPART_SIZE  = 8192   # must match transmitter
PACKET_SIZE   = 256

# Serial framing
FRAME_SYNC    = bytes([0xBB, 0x44])
ACK_BYTE      = 0x06
NACK_BYTE     = 0x15
CALIB_DONE    = 0xCA
CMD_CALIBRATE = 0xC2
CMD_RESET_ESP = 0xC3
CMD_SET_PRE   = 0xC1

# TX backend feedback URL
TX_FEEDBACK   = "http://localhost:5001/surface_feedback"

# Output dir
OUTPUT_DIR    = "./received_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4-FSK surface base frequencies
SURFACE_BASE = {
    "glass"    : 2000,
    "hardwood" : 1500,
    "softwood" : 1000,
    "metal"    : 3500,
    "unknown"  : 2000,
}

# ── Runtime state ─────────────────────────────────────────────────────────────
STATE = {
    "listening"       : False,
    "receiving"       : False,
    "serial_port"     : None,
    "cnn_model"       : None,
    "surface"         : "unknown",
    "base_freq"       : 2000,
    "freqs"           : [2000, 2750, 3500, 4250, 5000],
    "noise_floor"     : 0.0,
    "ecc_corrections" : 0,
    "aes_password"    : "SurfaceSyncSecretKey2025",
}

# Thread-safe ADC queue (numpy float32 arrays)
_adc_queue: queue.Queue = queue.Queue(maxsize=8192)

# Sub-part store: { sub_id -> { 'iv': bytes, 'packets': {pkt_id: bytes}, 'total': int, 'sub_total': int } }
_subpart_store: dict = {}
_file_parts: dict    = {}   # { sub_id -> plaintext bytes }

# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO") -> None:
    socketio.emit("log", {"message": msg, "level": level})
    print(f"[RX-{level}] {msg}")

# ── CRC-16/IBM ────────────────────────────────────────────────────────────────
def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else crc >> 1
    return crc & 0xFFFF

# ── AES-256-CBC decryption ────────────────────────────────────────────────────
def derive_key(password: str) -> bytes:
    return SHA256.new(password.encode()).digest()

def decrypt_subpart(iv: bytes, ciphertext: bytes, key: bytes) -> bytes | None:
    try:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ciphertext), AES.block_size)
    except (ValueError, KeyError):
        return None

# ── Frequency helpers ─────────────────────────────────────────────────────────
def compute_freqs(base_freq: int) -> list[int]:
    return [base_freq + i * FREQ_SPACING for i in range(5)]

# ══════════════════════════════════════════════════════════════════════════════
#  1D-CNN NOISE FILTER
# ══════════════════════════════════════════════════════════════════════════════
def build_cnn_model():
    """Binary 1D-CNN: signal window (1) vs noise window (0)."""
    model = keras.Sequential([
        keras.layers.Input(shape=(CNN_WINDOW, 1)),
        keras.layers.Conv1D(32, 7, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 5, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 3, padding="same", activation="relu"),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="SurfaceSync_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    weights_path = os.path.join(OUTPUT_DIR, "cnn_weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[RX] CNN weights loaded from {weights_path}")
    else:
        print("[RX] No CNN weights found - model is untrained (using bandpass fallback).")
    return model


def bandpass(samples: np.ndarray, low: float, high: float) -> np.ndarray:
    nyq = SAMPLE_RATE / 2.0
    sos = butter(5, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, samples)


def apply_noise_filter(samples: np.ndarray, model) -> np.ndarray:
    """
    Stage 1: Butterworth bandpass 800 Hz -> 8000 Hz (cover all surfaces).
    Stage 2: CNN sliding window mask (if model is trained).
    """
    pre = bandpass(samples, 800.0, 8000.0)

    if not TF_AVAILABLE or model is None:
        return pre

    # CNN window masking
    output = np.zeros_like(pre)
    for start in range(0, len(pre) - CNN_WINDOW, CNN_STRIDE):
        chunk = pre[start : start + CNN_WINDOW].astype(np.float32)
        norm  = chunk / (np.max(np.abs(chunk)) + 1e-9)
        x     = norm.reshape(1, CNN_WINDOW, 1)
        prob  = float(model.predict(x, verbose=0)[0][0])
        if prob >= CNN_THRESHOLD:
            output[start : start + CNN_WINDOW] += chunk * prob

    return output

# ══════════════════════════════════════════════════════════════════════════════
#  4-FSK DEMODULATOR
# ══════════════════════════════════════════════════════════════════════════════

def detect_symbol(window: np.ndarray, freqs: list[int]) -> int:
    """
    Given a SPS-sample window, return the dibit (0-3) whose frequency
    has the highest energy.  Uses zero-padded FFT for fine resolution.
    FFT resolution with FFT_PAD=4096: 44100/4096 ~= 10.8 Hz per bin.
    """
    fft_vals  = np.abs(np.fft.rfft(window, n=FFT_PAD))
    fft_freqs = np.fft.rfftfreq(FFT_PAD, d=1.0 / SAMPLE_RATE)
    tol       = FREQ_SPACING // 3   # 250 Hz tolerance (well inside 750 Hz gap)

    energies = []
    for f in freqs[:4]:
        mask = (fft_freqs >= f - tol) & (fft_freqs <= f + tol)
        energies.append(float(np.sum(fft_vals[mask])) if mask.any() else 0.0)

    return int(np.argmax(energies))


def demodulate_stream(samples: np.ndarray, freqs: list[int]) -> bytes:
    """
    Slide SPS-sample windows through *samples*, detect dibits, assemble bytes.
    Returns raw byte stream (may contain noise bytes before/after actual data).
    """
    dibits = []
    for start in range(0, len(samples) - SPS, SPS):
        window = samples[start : start + SPS]
        dibits.append(detect_symbol(window, freqs))

    # 4 dibits per byte, MSB first
    result = bytearray()
    for i in range(0, len(dibits) - 3, 4):
        b = (dibits[i]   << 6) | (dibits[i+1] << 4) | \
            (dibits[i+2] << 2) | (dibits[i+3])
        result.append(b & 0xFF)

    return bytes(result)

# ══════════════════════════════════════════════════════════════════════════════
#  SURFACE SWEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sweep(samples: np.ndarray) -> tuple[str, int]:
    """
    FFT analysis of the transmitter's sweep to identify surface and best freq.
    Returns (surface_label, base_freq_hz).
    """
    fft_vals  = np.abs(np.fft.rfft(samples))
    fft_freqs = np.fft.rfftfreq(len(samples), d=1.0 / SAMPLE_RATE)

    mask = (fft_freqs >= 1000) & (fft_freqs <= 7000)
    if not mask.any():
        return "unknown", 2000

    sub_vals  = fft_vals[mask]
    sub_freqs = fft_freqs[mask]
    peak_idx  = int(np.argmax(sub_vals))
    peak_freq = int(sub_freqs[peak_idx])
    sharpness = float(sub_vals[peak_idx]) / (float(np.mean(sub_vals)) + 1e-9)

    # Map sharpness to surface type and canonical base frequency
    if sharpness > 8.0:
        surface, base = "glass",    2000
    elif sharpness > 4.0:
        surface, base = "hardwood", 1500
    elif sharpness > 2.0:
        surface, base = "softwood", 1000
    else:
        # Metal: resonance at higher frequencies
        surface = "metal" if peak_freq > 3000 else "softwood"
        base    = 3500    if surface == "metal" else 1000

    log(f"Surface sweep: peak={peak_freq} Hz sharpness={sharpness:.1f} -> {surface}", "INFO")
    return surface, base


def report_to_transmitter(surface: str, base_freq: int) -> None:
    try:
        r = requests.post(TX_FEEDBACK,
                          json={"surface": surface, "base_freq": base_freq},
                          timeout=5)
        log(f"Sweep result sent to TX backend: {surface} @ {base_freq} Hz (HTTP {r.status_code})",
            "SUCCESS")
    except requests.RequestException as e:
        log(f"Could not reach TX backend: {e}", "ERROR")

# ══════════════════════════════════════════════════════════════════════════════
#  PACKET PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_packet(byte_stream: bytes) -> dict | None:
    """
    Scan *byte_stream* for acoustic sync word 0xAA 0x55 and parse packet.

    Returns dict with keys:
        sub_id, sub_total, pkt_id, pkt_total, flags,
        iv (bytes|None), chunk (bytes), valid (bool)
    or None if no sync found.
    """
    idx = byte_stream.find(b"\xAA\x55")
    if idx == -1:
        return None

    body_start = idx + 2   # after sync word
    # Need at least 11 bytes of header: sub_id(2)+sub_total(2)+pkt_id(2)+pkt_total(2)+flags(1)+data_len(2)
    if len(byte_stream) < body_start + 11 + 2:
        return None

    ptr = body_start
    sub_id    = struct.unpack_from(">H", byte_stream, ptr)[0]; ptr += 2
    sub_total = struct.unpack_from(">H", byte_stream, ptr)[0]; ptr += 2
    pkt_id    = struct.unpack_from(">H", byte_stream, ptr)[0]; ptr += 2
    pkt_total = struct.unpack_from(">H", byte_stream, ptr)[0]; ptr += 2
    flags     = byte_stream[ptr]; ptr += 1
    data_len  = struct.unpack_from(">H", byte_stream, ptr)[0]; ptr += 2

    if len(byte_stream) < ptr + data_len + 2:
        return None

    has_iv = bool(flags & 0x01)
    iv     = None
    chunk  = b""

    if has_iv:
        if data_len < 16:
            return None
        iv    = byte_stream[ptr : ptr + 16]
        chunk = byte_stream[ptr + 16 : ptr + data_len]
    else:
        chunk = byte_stream[ptr : ptr + data_len]

    ptr += data_len

    # Verify CRC-16 over body (from sub_id to end of data)
    body    = byte_stream[body_start : ptr]
    rx_crc  = struct.unpack_from(">H", byte_stream, ptr)[0]
    calc    = crc16(body)

    return {
        "sub_id"    : sub_id,
        "sub_total" : sub_total,
        "pkt_id"    : pkt_id,
        "pkt_total" : pkt_total,
        "flags"     : flags,
        "iv"        : iv,
        "chunk"     : chunk,
        "valid"     : (calc == rx_crc),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  SUB-PART STORE + FILE REASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def store_packet(pkt: dict) -> None:
    sid = pkt["sub_id"]
    if sid not in _subpart_store:
        _subpart_store[sid] = {
            "total"     : pkt["pkt_total"],
            "sub_total" : pkt["sub_total"],
            "packets"   : {},
            "iv"        : None,
        }
    entry = _subpart_store[sid]
    if pkt["iv"] is not None:
        entry["iv"] = pkt["iv"]
    entry["packets"][pkt["pkt_id"]] = pkt["chunk"]


def try_decrypt_subpart(sub_id: int, key: bytes) -> bool:
    """
    Attempt to decrypt sub-part *sub_id* when all its packets are present.
    Stores result in _file_parts and returns True on success.
    """
    entry = _subpart_store.get(sub_id)
    if entry is None:
        return False
    if len(entry["packets"]) < entry["total"]:
        return False
    if entry["iv"] is None:
        return False

    ciphertext = b"".join(entry["packets"][i]
                          for i in range(entry["total"]))
    plaintext  = decrypt_subpart(entry["iv"], ciphertext, key)
    if plaintext is None:
        log(f"Decryption failed for sub-part {sub_id}", "ERROR")
        return False

    _file_parts[sub_id] = plaintext
    log(f"Sub-part {sub_id+1}/{entry['sub_total']} decrypted "
        f"({len(plaintext):,} B)", "SUCCESS")
    socketio.emit("subpart_done", {
        "sub_id"    : sub_id,
        "sub_total" : entry["sub_total"],
        "size"      : len(plaintext),
    })
    return True


def try_reassemble_file() -> bytes | None:
    """Return assembled file bytes when all sub-parts are present."""
    if not _file_parts:
        return None
    sub_total = next(iter(_subpart_store.values()))["sub_total"]
    if len(_file_parts) < sub_total:
        return None
    return b"".join(_file_parts[i] for i in range(sub_total))


def save_file(data: bytes) -> str:
    if data[:2]   == b"\xFF\xD8": ext = ".jpg"
    elif data[:4] == b"\x89PNG":  ext = ".png"
    elif data[:4] == b"GIF8":     ext = ".gif"
    elif data[:4] == b"%PDF":     ext = ".pdf"
    elif all(b < 128 for b in data[:64]): ext = ".txt"
    else:                                  ext = ".bin"

    fname = f"received_{int(time.time())}{ext}"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(data)
    return fpath

# ══════════════════════════════════════════════════════════════════════════════
#  SERIAL READER THREAD  (ESP32-WROOM -> Python)
# ══════════════════════════════════════════════════════════════════════════════

def _serial_reader(ser: serial.Serial) -> None:
    buf = bytearray()
    log("Serial reader started.", "INFO")

    while STATE["listening"]:
        raw = ser.read(ser.in_waiting or 1)
        if not raw:
            continue
        buf.extend(raw)

        # Parse frames
        while len(buf) >= 6:
            idx = buf.find(FRAME_SYNC)
            if idx == -1:
                buf = buf[-2:]
                break
            if idx > 0:
                # Check for single-byte commands mixed in
                for b in buf[:idx]:
                    if b == CALIB_DONE:
                        log("ESP32 calibration complete.", "SUCCESS")
                buf = buf[idx:]
                continue

            if len(buf) < 4:
                break

            len_word  = (buf[2] << 8) | buf[3]
            has_pre   = bool(len_word & 0x8000)
            data_len  = len_word & 0x7FFF
            total_len = 4 + data_len + 2

            if len(buf) < total_len:
                break

            frame_data = buf[4 : 4 + data_len]
            frame_crc  = (buf[4 + data_len] << 8) | buf[4 + data_len + 1]
            buf        = buf[total_len:]

            if crc16(frame_data) != frame_crc:
                ser.write(bytes([NACK_BYTE]))
                STATE["ecc_corrections"] += 1
                continue

            ser.write(bytes([ACK_BYTE]))

            # Decode int16 big-endian samples
            n_samps = data_len // 2
            samples = np.frombuffer(frame_data, dtype=">i2").astype(np.float32)

            try:
                _adc_queue.put_nowait((samples, has_pre))
            except queue.Full:
                pass  # drop oldest; prefer freshness

    log("Serial reader stopped.", "INFO")

# ══════════════════════════════════════════════════════════════════════════════
#  DSP PIPELINE THREAD
# ══════════════════════════════════════════════════════════════════════════════

def _dsp_pipeline() -> None:
    """
    Drains _adc_queue, denoises, demodulates, parses packets, reassembles.
    State machine: IDLE -> CALIBRATING -> LISTENING -> RECEIVING -> DONE
    """
    model    = STATE.get("cnn_model")
    aes_key  = derive_key(STATE["aes_password"])
    freqs    = STATE["freqs"]

    accumulator   = np.array([], dtype=np.float32)  # rolling sample buffer
    sweep_buf     = np.array([], dtype=np.float32)   # buffer during sweep
    PROCESS_BLOCK = SAMPLE_RATE // 10                # 100 ms blocks = 4410 samples
    SWEEP_SECS    = 8                                # seconds to collect sweep audio

    pipeline_state  = "LISTENING"
    sweep_collecting = False
    sweep_deadline   = 0.0
    preamble_seen    = False

    log("DSP pipeline started.", "INFO")

    while STATE["listening"]:
        try:
            samples, has_pre = _adc_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        accumulator = np.concatenate([accumulator, samples])

        # ── Preamble gate: ESP32 flagged hardware preamble energy ────────────
        if has_pre and not STATE["receiving"]:
            preamble_seen = True
            STATE["receiving"] = True
            socketio.emit("status", {"state": "receiving"})
            log("Preamble detected (hardware gate) - demodulation active.", "SUCCESS")

        # ── Process in PROCESS_BLOCK chunks ──────────────────────────────────
        while len(accumulator) >= PROCESS_BLOCK:
            block       = accumulator[:PROCESS_BLOCK]
            accumulator = accumulator[PROCESS_BLOCK:]

            # Raw waveform snapshot to UI
            socketio.emit("waveform_raw",
                          {"samples": block[::8].tolist()})

            # Noise filter
            clean = apply_noise_filter(block, model)
            socketio.emit("waveform_clean",
                          {"samples": clean[::8].tolist()})

            # SNR estimate
            sig_rms   = float(np.sqrt(np.mean(clean ** 2)))
            noise_rms = STATE["noise_floor"] + 1e-9
            snr       = 20.0 * np.log10(sig_rms / noise_rms)

            # Dominant frequency in FSK band
            fft_v = np.abs(np.fft.rfft(clean))
            fft_f = np.fft.rfftfreq(len(clean), d=1.0 / SAMPLE_RATE)
            band  = (fft_f >= 800) & (fft_f <= 8000)
            dom_f = int(fft_f[band][np.argmax(fft_v[band])]) if band.any() else 0

            socketio.emit("signal_stats", {
                "snr_db"     : round(snr, 1),
                "freq_hz"    : dom_f,
                "signal_pct" : min(100, max(0, int((snr + 20) * 2.5))),
                "ecc"        : STATE["ecc_corrections"],
            })

            # Sweep collection
            if sweep_collecting:
                sweep_buf = np.concatenate([sweep_buf, clean])
                if time.time() > sweep_deadline:
                    sweep_collecting = False
                    _process_sweep(sweep_buf)
                    sweep_buf = np.array([], dtype=np.float32)
                continue

            # Only demodulate when preamble has been seen
            if not preamble_seen:
                continue

            # Demodulate block -> raw bytes
            raw_bytes = demodulate_stream(clean, freqs)
            if len(raw_bytes) < 16:
                continue

            # Parse packet
            pkt = parse_packet(raw_bytes)
            if pkt is None:
                continue

            if not pkt["valid"]:
                STATE["ecc_corrections"] += 1
                socketio.emit("ecc", {"count": STATE["ecc_corrections"]})
                log(f"CRC fail sub={pkt.get('sub_id','?')} pkt={pkt.get('pkt_id','?')}", "ERROR")
                continue

            # Store packet
            store_packet(pkt)
            sub_id    = pkt["sub_id"]
            sub_total = pkt["sub_total"]

            # Update progress
            received_total = sum(len(e["packets"])
                                 for e in _subpart_store.values())
            expected_total = sum(e["total"]
                                 for e in _subpart_store.values()
                                 if e["total"])
            if expected_total > 0:
                progress = (received_total / expected_total) * 100.0
                socketio.emit("progress", {
                    "percent"  : round(progress, 1),
                    "received" : received_total,
                    "total"    : expected_total,
                })

            # Try decrypt this sub-part
            if try_decrypt_subpart(sub_id, aes_key):
                # Try full file reassembly
                file_data = try_reassemble_file()
                if file_data is not None:
                    preamble_seen    = False
                    STATE["receiving"] = False
                    fpath = save_file(file_data)
                    log(f"[OK] File reassembled: {os.path.basename(fpath)} "
                        f"({len(file_data):,} B)", "SUCCESS")
                    socketio.emit("status", {"state": "done"})
                    is_text = fpath.endswith(".txt")
                    socketio.emit("file_ready", {
                        "filename" : os.path.basename(fpath),
                        "size"     : len(file_data),
                        "is_text"  : is_text,
                        "text"     : file_data.decode("utf-8", errors="replace")
                                     if is_text else None,
                    })
                    # Reset stores for next transfer
                    _subpart_store.clear()
                    _file_parts.clear()
                    socketio.emit("status", {"state": "listening"})

    log("DSP pipeline stopped.", "INFO")


def _process_sweep(sweep_samples: np.ndarray) -> None:
    """Analyze sweep audio and report surface type to TX backend."""
    surface, base_freq = analyze_sweep(sweep_samples)
    STATE["surface"]   = surface
    STATE["base_freq"] = base_freq
    STATE["freqs"]     = compute_freqs(base_freq)

    socketio.emit("sweep_result", {
        "surface" : surface,
        "freqs"   : STATE["freqs"],
    })

    t = threading.Thread(
        target=report_to_transmitter,
        args=(surface, base_freq),
        daemon=True,
    )
    t.start()

# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return "<h2>SurfaceSync Receiver Backend - :5002</h2>"


@app.route("/start_listening", methods=["POST"])
def start_listening():
    if STATE["listening"]:
        return jsonify({"error": "Already listening"}), 409

    body = request.get_json(silent=True) or {}
    port = body.get("port")
    baud = int(body.get("baud", 921600))

    # Load CNN
    if TF_AVAILABLE and STATE["cnn_model"] is None:
        log("Loading CNN model ...", "INFO")
        STATE["cnn_model"] = build_cnn_model()

    # Open serial
    if port:
        try:
            ser = serial.Serial(port, baud, timeout=2.0)
            STATE["serial_port"] = ser
            log(f"Serial connected: {port} @ {baud}", "SUCCESS")
        except serial.SerialException as e:
            return jsonify({"error": str(e)}), 500
    else:
        log("No serial port - simulation mode.", "INFO")

    STATE["listening"] = True
    socketio.emit("status", {"state": "listening"})

    if STATE.get("serial_port"):
        threading.Thread(
            target=_serial_reader,
            args=(STATE["serial_port"],),
            daemon=True,
        ).start()

    threading.Thread(target=_dsp_pipeline, daemon=True).start()
    log("Receiver listening for surface vibrations.", "SUCCESS")
    return jsonify({"status": "listening"}), 200


@app.route("/stop_listening", methods=["POST"])
def stop_listening():
    STATE["listening"]  = False
    STATE["receiving"]  = False
    socketio.emit("status", {"state": "idle"})
    ser = STATE.get("serial_port")
    if ser and ser.is_open:
        ser.close()
    log("Receiver stopped.", "INFO")
    return jsonify({"status": "stopped"}), 200


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """
    5-second noise floor calibration.
    Tells ESP32-WROOM to measure noise; result fed back via serial 0xCA byte.
    Also collects local ADC data to estimate noise_floor.
    """
    if not STATE["listening"]:
        return jsonify({"error": "Call /start_listening first"}), 400

    def _cal():
        target   = SAMPLE_RATE * 5
        collected = np.array([], dtype=np.float32)
        deadline  = time.time() + 6

        # Tell ESP32 to start calibration
        ser = STATE.get("serial_port")
        if ser and ser.is_open:
            ser.write(bytes([CMD_CALIBRATE]))

        while len(collected) < target and time.time() < deadline:
            try:
                samples, _ = _adc_queue.get(timeout=0.1)
                collected = np.concatenate([collected, samples])
            except queue.Empty:
                pass

        if len(collected) > 0:
            rms = float(np.sqrt(np.mean(collected ** 2)))
            STATE["noise_floor"] = rms
            log(f"Noise floor: {rms:.4f} RMS", "INFO")
            socketio.emit("calibration_done", {"noise_floor": rms})

    threading.Thread(target=_cal, daemon=True).start()
    return jsonify({"status": "calibrating"}), 202


@app.route("/inject_samples", methods=["POST"])
def inject_samples():
    """Hardware-free test: POST JSON { samples: [float, ...] }"""
    body    = request.get_json(force=True)
    samples = np.array(body.get("samples", []), dtype=np.float32)
    if len(samples) == 0:
        return jsonify({"error": "Empty samples"}), 400
    _adc_queue.put_nowait((samples, False))
    return jsonify({"queued": len(samples)}), 200


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    fpath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(fpath):
        return jsonify({"error": "Not found"}), 404
    return send_file(fpath, as_attachment=True)


@app.route("/train_cnn", methods=["POST"])
def train_cnn():
    """
    Online retraining.
    JSON: { "signal_samples": [[512 floats], ...], "noise_samples": [[512 floats], ...] }
    """
    if not TF_AVAILABLE or STATE["cnn_model"] is None:
        return jsonify({"error": "TensorFlow not available"}), 503

    body = request.get_json(force=True)
    sig  = np.array(body.get("signal_samples", []), dtype=np.float32)
    nse  = np.array(body.get("noise_samples",  []), dtype=np.float32)

    if len(sig) == 0 or len(nse) == 0:
        return jsonify({"error": "Need signal and noise samples"}), 400

    X = np.concatenate([sig, nse]).reshape(-1, CNN_WINDOW, 1)
    y = np.concatenate([np.ones(len(sig)), np.zeros(len(nse))])

    model = STATE["cnn_model"]
    hist  = model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    ckpt = os.path.join(OUTPUT_DIR, "cnn_weights.h5")
    model.save_weights(ckpt)
    final_loss = float(hist.history["loss"][-1])
    log(f"CNN retrained on {len(X)} windows. Loss: {final_loss:.4f}", "SUCCESS")
    return jsonify({"status": "trained", "final_loss": final_loss}), 200


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
        "listening"       : STATE["listening"],
        "receiving"       : STATE["receiving"],
        "surface"         : STATE["surface"],
        "base_freq"       : STATE["base_freq"],
        "freqs"           : STATE["freqs"],
        "noise_floor"     : round(STATE["noise_floor"], 6),
        "ecc_corrections" : STATE["ecc_corrections"],
        "subparts_stored" : len(_subpart_store),
        "file_parts_done" : len(_file_parts),
        "serial_open"     : bool(STATE.get("serial_port") and STATE["serial_port"].is_open),
        "cnn_loaded"      : STATE["cnn_model"] is not None,
    }), 200


# ── Socket.IO ─────────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    emit("status", {
        "state"   : "receiving" if STATE["receiving"] else
                    "listening" if STATE["listening"] else "idle",
        "surface" : STATE["surface"],
        "freqs"   : STATE["freqs"],
    })

@socketio.on("ping_backend")
def on_ping():
    emit("pong", {"ts": time.time()})

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SurfaceSync - Receiver Backend")
    print("  http://0.0.0.0:5002")
    print("  POST /start_listening  -> begin ADC pipeline")
    print("  POST /stop_listening   -> halt")
    print("  POST /calibrate        -> 5-sec noise calibration")
    print("  POST /inject_samples   -> simulation (no hardware)")
    print("  POST /train_cnn        -> online CNN retraining")
    print("  GET  /download/<file>  -> retrieve received file")
    print("  GET  /status           -> pipeline state")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5002, debug=False)