import numpy as np
import random

class LSB_Sub:
    @staticmethod
    def embed(image, message, key):
        # 1. Chuyển tin nhắn sang Byte và thêm 3 byte NULL để đánh dấu kết thúc cực an toàn
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        h, w = image.shape
        total_pixels = h * w
        
        # 2. Tính số bit nhúng mỗi pixel
        n_bits = int(np.ceil(len(bin_msg) / total_pixels))
        n_bits = min(max(n_bits, 1), 4)

        # 3. QUAN TRỌNG: Đệm thêm bit '0' để độ dài bin_msg chia hết cho n_bits
        # Điều này giúp pixel cuối cùng nhận đủ n_bits, không bị mất dữ liệu
        while len(bin_msg) % n_bits != 0:
            bin_msg += '0'

        # 4. Sử dụng Random độc lập cho tọa độ
        rng = random.Random(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(coords)

        stego = image.copy().astype(np.uint8)
        idx = 0
        
        for r, c in coords:
            if idx >= len(bin_msg): break
            
            # Lấy đúng n_bits
            bits = bin_msg[idx : idx + n_bits]
            actual_n = len(bits)
            
            val = int(stego[r, c])
            # Tạo mask để xóa n bits cũ
            mask = (0xFF << actual_n) & 0xFF
            # Lấy giá trị bit mới
            bit_val = int(bits, 2)
            
            # Thay thế bit (Substitution)
            stego[r, c] = (val & mask) | bit_val
            idx += actual_n
            
        return stego, n_bits

    @staticmethod
    def extract(stego, key, n_bits=1):
        h, w = stego.shape
        # Phải dùng cùng một Random Seed và cùng cách Shuffle
        rng = random.Random(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(coords)
        
        bin_msg = ""
        # 1. Thu thập toàn bộ bit từ các pixel đã nhúng
        for r, c in coords:
            val = int(stego[r, c])
            # Trích xuất n_bits cuối
            bin_msg += format(val & ((1 << n_bits) - 1), f'0{n_bits}b')
            
            # Kiểm tra dấu hiệu kết thúc (3 bytes NULL liên tiếp = 24 bit 0)
            if len(bin_msg) % 8 == 0 and "000000000000000000000000" in bin_msg[-48:]:
                break
        
        # 2. Chuyển bit sang mảng Byte
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: break
            
            byte_val = int(byte_str, 2)
            if byte_val == 0: # Dừng lại ngay khi gặp byte NULL đầu tiên
                break
            byte_list.append(byte_val)
            
        # 3. Giải mã UTF-8 (errors='ignore' để tránh crash nếu có bit nhiễu)
        try:
            return bytes(byte_list).decode('utf-8')
        except Exception as e:
            return bytes(byte_list).decode('utf-8', errors='ignore')