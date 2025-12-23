import numpy as np
from core.pvd import PVD

class HistogramShifting:
    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        # Tính toán histogram chuẩn
        hist_data = np.histogram(image, bins=256, range=(0, 256))[0]
        peak = int(np.argmax(hist_data))
        
        stego = image.copy().astype(np.int32)
        # Shift các pixel từ peak + 1 sang phải
        stego[image > peak] += 1
        
        msg_idx = 0
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if image[r, c] == peak and msg_idx < len(bin_msg):
                    if bin_msg[msg_idx] == '1':
                        stego[r, c] = peak + 1
                    msg_idx += 1
        return np.clip(stego, 0, 255).astype(np.uint8), peak

    @staticmethod
    def extract(stego, key, peak):
        bin_msg = ""
        flat = stego.flatten()
        for val in flat:
            if val == peak:
                bin_msg += "0"
            elif val == peak + 1:
                bin_msg += "1"
            if len(bin_msg) % 8 == 0 and "00000000" in bin_msg[-16:]:
                break
        return PVD.bin_to_msg(bin_msg)