import numpy as np
import random

class LSB_Sub:
    @staticmethod
    def embed(image, message, key):
        # 1. Chuyển tin nhắn sang chuỗi bit
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        data_len = len(bin_msg)
        
        h, w = image.shape
        total_pixels = h * w
        
        # 2. TÍNH TOÁN SỐ BIT CẦN DÙNG TRÊN MỖI PIXEL (n_bits)
        # Ví dụ: Nếu data_len = 150.000 bits, total_pixels = 65.536
        # n_bits = ceil(150.000 / 65.536) = 3 bits/pixel
        n_bits = int(np.ceil(data_len / total_pixels))
        
        if n_bits > 4:
            print("Cảnh báo: Dữ liệu quá lớn, vượt quá 4-bit LSB. Sẽ bị cắt bớt.")
            n_bits = 4
            bin_msg = bin_msg[:total_pixels * n_bits]
            data_len = len(bin_msg)

        # 3. Xáo trộn vị trí pixel
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        random.seed(key)
        random.shuffle(all_coords)

        stego_img = image.copy().astype(np.uint8)
        msg_idx = 0
        
        # 4. Nhúng Multi-bit
        for r, c in all_coords:
            if msg_idx >= data_len:
                break
            
            # Lấy n bit tiếp theo từ tin nhắn
            bits_to_embed = bin_msg[msg_idx : msg_idx + n_bits]
            actual_bits = len(bits_to_embed)
            
            pixel_val = int(stego_img[r, c])
            
            # Tạo mask để xóa n bit cuối của pixel
            # Ví dụ n=3: mask = 11111000 (248)
            mask = (0xFF << actual_bits) & 0xFF
            
            # Chuyển đoạn bit tin nhắn sang số nguyên
            bit_val = int(bits_to_embed, 2)
            
            # Nhúng vào pixel
            new_pixel_val = (pixel_val & mask) | bit_val
            stego_img[r, c] = np.uint8(new_pixel_val)
            
            msg_idx += actual_bits
            
        return stego_img, n_bits # Trả về thêm n_bits để biết đường mà trích xuất
    
    @staticmethod
    def extract(stego_image, key, n_bits=1):
        h, w = stego_image.shape
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        import random
        random.seed(key)
        random.shuffle(all_coords)
        
        bin_msg = ""
        # 1. Đọc đủ bit để có thể chứa tin nhắn (Duyệt hết pixel)
        for r, c in all_coords:
            pixel_val = int(stego_image[r, c])
            # Lấy n_bits cuối
            extracted_bits = format(pixel_val & ((1 << n_bits) - 1), f'0{n_bits}b')
            bin_msg += extracted_bits
            
            # Kiểm tra nhanh: Nếu có 16 bit 0 liên tiếp (dấu hiệu kết thúc an toàn) thì dừng
            if len(bin_msg) > 16 and "0000000000000000" in bin_msg[-32:]:
                break
        
        # 2. Chuyển bit sang ký tự theo từng cụm 8 bit
        chars = ""
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: break
            
            char_code = int(byte_str, 2)
            if char_code == 0: # Gặp ký tự NULL (\0) thì dừng hẳn
                break
            chars += chr(char_code)
            
        return chars
    