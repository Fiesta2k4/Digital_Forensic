import numpy as np

class DifferenceExpansion:
    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        msg_idx = 0
        layers = 0
        h, w = image.shape
        stego = image.copy().astype(np.int32)

        while msg_idx < len(bin_msg) and layers < 3:
            flat_stego = stego.flatten()
        
            # Thuật toán sẽ tự động dừng khi hết pixel hoặc hết tin nhắn
            for i in range(0, len(flat_stego) - 1, 2):
                if msg_idx >= len(bin_msg):
                    break
                
                x, y = int(flat_stego[i]), int(flat_stego[i+1])
                l = (x + y) // 2
                d = x - y
                
                b = int(bin_msg[msg_idx])
                d_new = 2 * d + b
                
                x_new = l + (d_new + 1) // 2
                y_new = l - d_new // 2
                
                if 0 <= x_new <= 255 and 0 <= y_new <= 255:
                    flat_stego[i], flat_stego[i+1] = x_new, y_new
                    msg_idx += 1
                # Nếu không nhúng được do tràn, thuật toán sẽ bỏ qua cặp đó và tìm cặp tiếp theo
            layers += 1 

        return flat_stego.reshape(image.shape).astype(np.uint8), layers

    @staticmethod
    def extract(stego_image, key, layers=1):
        # Lưu ý: Với DE, chúng ta cần duyệt theo cặp pixel
        flat = stego_image.flatten().astype(np.int32)
        bin_msg = ""
        
        for i in range(0, len(flat) - 1, 2):
            x, y = flat[i], flat[i+1]
            d_new = x - y
            # Bit nhúng là LSB của hiệu số (abs để tránh số âm)
            bin_msg += str(abs(d_new) % 2)
            
            if len(bin_msg) % 8 == 0 and "00000000" in bin_msg[-16:]:
                break
                
        chars = ""
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: break
            code = int(byte_str, 2)
            if code == 0: break
            chars += chr(code)
        return chars