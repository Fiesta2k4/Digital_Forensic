import numpy as np
import random

class LSB_Matching:
    @staticmethod
    def embed(image, message, key):
        # 1. Chuyển tin nhắn sang bit
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        h, w = image.shape
        total_pixels = h * w
        
        # 2. Tính số bit cần nhúng mỗi pixel (n_bits)
        n_bits = int(np.ceil(len(bin_msg) / total_pixels))
        n_bits = min(max(n_bits, 1), 4) # Giới hạn 1-4 bits
        
        # 3. Tạo bộ sinh số ngẫu nhiên RIÊNG BIỆT cho tọa độ
        # Để không bị ảnh hưởng bởi việc gọi random.choice bên dưới
        rng_coords = random.Random(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng_coords.shuffle(coords)
        
        # 4. Tạo bộ sinh số ngẫu nhiên cho việc Matching (+/- 1)
        rng_match = random.Random(key)
        
        stego = image.copy().astype(np.int16)
        msg_idx = 0
        mod_val = 2**n_bits

        for r, c in coords:
            if msg_idx >= len(bin_msg): break
            
            # Lấy đoạn bit mục tiêu
            bits = bin_msg[msg_idx : msg_idx + n_bits]
            target_val = int(bits.ljust(n_bits, '0'), 2)
            
            # Giá trị hiện tại của pixel theo modulo
            current_val = stego[r, c] % mod_val
            
            if current_val != target_val:
                # LỖI BIÊN: Xử lý 0 và 255
                if stego[r, c] == 0:
                    stego[r, c] += 1
                elif stego[r, c] == 255:
                    stego[r, c] -= 1
                else:
                    # Cộng hoặc trừ 1 ngẫu nhiên để khớp bit
                    stego[r, c] += rng_match.choice([1, -1])
            
            # Đảm bảo sau khi +/- 1, bit cuối vẫn khớp (Double check)
            # Nếu vẫn không khớp do lỗi toán học, ta ép bằng LSB Sub
            if stego[r, c] % mod_val != target_val:
                stego[r, c] = (stego[r, c] // mod_val) * mod_val + target_val
            
            stego[r, c] = np.clip(stego[r, c], 0, 255)
            msg_idx += len(bits)
            
        return stego.astype(np.uint8), n_bits

    @staticmethod
    def extract(stego, key, n_bits=1):
        h, w = stego.shape
        rng_coords = random.Random(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng_coords.shuffle(coords)
        
        mod_val = 2**n_bits
        bin_msg = ""
        
        for r, c in coords:
            val = int(stego[r, c]) % mod_val
            # Chuyển giá trị lấy được sang chuỗi bit độ dài n_bits
            bin_msg += format(val, f'0{n_bits}b')
            
            # Kiểm tra ký tự kết thúc (8 bit 0)
            if len(bin_msg) % 8 == 0 and "00000000" in bin_msg[-16:]:
                break
        
        # Chuyển bit sang chữ
        chars = ""
        for i in range(0, len(bin_msg), 8):
            byte = bin_msg[i:i+8]
            if len(byte) < 8: break
            code = int(byte, 2)
            if code == 0: break
            chars += chr(code)
            
        return chars