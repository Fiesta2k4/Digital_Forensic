import numpy as np
import random

class LSB_Sub:
    @staticmethod
    def message_to_bin(message):
        # Chuyển string sang chuỗi bit (string)
        return ''.join([format(ord(i), "08b") for i in message])

    @staticmethod
    def bin_to_message(binary_data):
        all_bytes = [binary_data[i: i + 8] for i in range(0, len(binary_data), 8)]
        decoded_data = ""
        for byte in all_bytes:
            if len(byte) < 8: break
            decoded_data += chr(int(byte, 2))
        return decoded_data

    @staticmethod
    def embed(image, message, key):
        # 1. Chuẩn bị chuỗi bit (String)
        bin_msg = LSB_Sub.message_to_bin(message) + "00000000" # NULL terminator
        data_len = len(bin_msg)
        
        h, w = image.shape
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        
        # 2. Xáo trộn vị trí dựa trên Khóa K (Seed)
        random.seed(key)
        random.shuffle(all_coords)
        
        if data_len > len(all_coords):
            raise ValueError(f"Tin nhắn quá dài! Cần {data_len} pixel, ảnh có {len(all_coords)} pixel.")

        stego_img = image.copy().astype(np.uint8)
        
        # 3. Nhúng bit
        for i in range(data_len):
            r, c = all_coords[i]
            # Đảm bảo lấy giá trị số nguyên
            pixel_val = int(stego_img[r, c])
            bit = int(bin_msg[i]) # Chuyển ký tự '0'/'1' sang số 0/1
            
            # LSB Substitution
            new_val = (pixel_val & ~1) | bit
            stego_img[r, c] = np.uint8(new_val)
            
        return stego_img

    @staticmethod
    def extract(stego_image, key):
        h, w = stego_image.shape
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        
        random.seed(key)
        random.shuffle(all_coords)
        
        bin_msg = ""
        for r, c in all_coords:
            pixel_val = int(stego_image[r, c])
            bin_msg += str(pixel_val & 1)
            
            if len(bin_msg) >= 8 and bin_msg[-8:] == "00000000":
                break
        
        return LSB_Sub.bin_to_message(bin_msg[:-8])