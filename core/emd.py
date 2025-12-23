import numpy as np
from core.lsb_sub import LSB_Sub
from core.pvd import PVD

class EMD:
    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        val = int(bin_msg, 2)
        digits = []
        while val > 0:
            digits.append(val % 5)
            val //= 5
            
        stego = image.flatten().astype(np.int32)
        for i in range(len(digits)):
            p1, p2 = stego[2*i], stego[2*i+1]
            f = (p1 * 1 + p2 * 2) % 5
            s = (digits[i] - f) % 5
            if s == 1: p1 += 1
            elif s == 2: p2 += 1
            elif s == 3: p2 -= 1
            elif s == 4: p1 -= 1
            stego[2*i], stego[2*i+1] = p1, p2
        return np.clip(stego.reshape(image.shape), 0, 255).astype(np.uint8), len(digits)

    @staticmethod
    def extract(stego, key, n_digits):
        flat = stego.flatten().astype(np.int32)
        val = 0
        power = 1
        for i in range(n_digits):
            p1, p2 = flat[2*i], flat[2*i+1]
            f = (p1 * 1 + p2 * 2) % 5
            val += f * power
            power *= 5
        bin_msg = bin(val)[2:]
        # Cần pad đủ 8 bit
        bin_msg = bin_msg.zfill(((len(bin_msg) + 7) // 8) * 8)
        return PVD.bin_to_msg(bin_msg)