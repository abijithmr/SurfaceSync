import transmitter_backend as tx
import receiver_backend as rx

raw = b'HelloSurfaceSync' * 30
packets = tx.build_packets(raw)
packets_enc = tx.encrypt_all_packets(packets, tx.CFG['AES_PASSWORD'])
frame = tx._build_wire_frame(packets_enc[0])
pkt = rx.parse_wire_frame(frame)
print('parsed valid:', pkt and pkt.get('valid'))
pt = rx.decrypt_packet(pkt['iv'], pkt['ciphertext'], rx.derive_key(rx.CFG['AES_PASSWORD']))
print('decrypted equals payload:', pt == packets[0]['payload'])
