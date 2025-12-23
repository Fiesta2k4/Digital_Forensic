import numpy as np

class PVD:
    ranges = [0, 8, 16, 32, 64, 128, 256]

    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        stego = image.flatten().astype(np.int32)
        idx = 0
        for i in range(0, len(stego)-1, 2):
            if idx >= len(bin_msg): break
            p1, p2 = stego[i], stego[i+1]
            d = abs(p1 - p2)
            for r in range(len(PVD.ranges)-1):
                if PVD.ranges[r] <= d < PVD.ranges[r+1]:
                    low, high = PVD.ranges[r], PVD.ranges[r+1]-1
                    t = int(np.log2(high - low + 1))
                    bits = bin_msg[idx : idx+t]
                    if not bits: break
                    b = int(bits.ljust(t, '0'), 2)
                    d_new = low + b
                    diff = d_new - d
                    if p1 >= p2:
                        stego[i] += int(np.ceil(diff/2))
                        stego[i+1] -= int(np.floor(diff/2))
                    else:
                        stego[i] -= int(np.floor(diff/2))
                        stego[i+1] += int(np.ceil(diff/2))
                    idx += len(bits)
                    break
        return np.clip(stego.reshape(image.shape), 0, 255).astype(np.uint8), "PVD-Adaptive"

    @staticmethod
    def extract(stego, key):
        flat = stego.flatten().astype(np.int32)
        bin_msg = ""
        for i in range(0, len(flat)-1, 2):
            p1, p2 = flat[i], flat[i+1]
            d = abs(p1 - p2)
            for r in range(len(PVD.ranges)-1):
                if PVD.ranges[r] <= d < PVD.ranges[r+1]:
                    low = PVD.ranges[r]
                    t = int(np.log2(PVD.ranges[r+1] - low))
                    b = d - low
                    bin_msg += format(b, f'0{t}b')
                    break
            if "00000000" in bin_msg[-16:]: break
        return PVD.bin_to_msg(bin_msg)

    @staticmethod
    def bin_to_msg(bin_data):
        chars = ""
        for i in range(0, len(bin_data), 8):
            byte = bin_data[i:i+8]
            if len(byte) < 8: break
            code = int(byte, 2)
            if code == 0: break
            chars += chr(code)
        return chars