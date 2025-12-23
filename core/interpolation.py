import numpy as np
import cv2

from core.pvd import PVD

class Interpolation:
    @staticmethod
    def embed(image, message, key):
        h, w = image.shape
        # Down-sample
        low_res = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        # Up-sample
        up_res = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
        
        diff = image.astype(np.int16) - up_res.astype(np.int16)
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        
        # Nhúng vào LSB của mảng sai số
        flat_diff = diff.flatten()
        for i in range(min(len(bin_msg), len(flat_diff))):
            bit = int(bin_msg[i])
            flat_diff[i] = (flat_diff[i] & ~1) | bit
            
        stego = up_res.astype(np.int16) + flat_diff.reshape(image.shape)
        return np.clip(stego, 0, 255).astype(np.uint8), len(bin_msg)

    @staticmethod
    def extract(stego, key, msg_len):
        h, w = stego.shape
        # Tái tạo lại ảnh up_res giống hệt lúc nhúng
        low_res = cv2.resize(stego, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
        up_res = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Sai số hiện tại chính là diff đã chứa bit
        diff = stego.astype(np.int16) - up_res.astype(np.int16)
        flat_diff = diff.flatten()
        
        bin_msg = ""
        for i in range(msg_len):
            bin_msg += str(abs(flat_diff[i]) % 2)
            
        return PVD.bin_to_msg(bin_msg)