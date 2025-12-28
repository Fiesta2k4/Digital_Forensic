import numpy as np

class HistogramShifting:
    @staticmethod
    def embed(image, message, key):
        # Chuyển message sang UTF-8 bytes và thêm 3 bytes NULL để đánh dấu kết thúc
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        # Tính toán histogram chuẩn
        hist_data = np.histogram(image, bins=256, range=(0, 256))[0]
        
        # Tìm peak an toàn (không phải 0 hoặc 255 để tránh overflow)
        # Nếu peak là 255, tìm peak tiếp theo
        peak_candidates = np.argsort(hist_data)[::-1]  # Sort theo thứ tự giảm dần
        peak = None
        
        for candidate in peak_candidates:
            if 0 < candidate < 255:  # Peak phải nằm trong khoảng [1, 254]
                peak = int(candidate)
                break
        
        if peak is None:
            # Fallback: nếu không tìm được peak an toàn, chọn peak có count cao nhất trong [1, 254]
            safe_hist = hist_data[1:255]
            if np.max(safe_hist) == 0:
                raise ValueError("Ảnh không phù hợp cho Histogram Shifting (không có pixel trong khoảng [1,254])")
            peak = int(np.argmax(safe_hist) + 1)  # +1 vì bỏ qua index 0
        
        # KIỂM TRA DUNG LƯỢNG
        available_pixels = hist_data[peak]
        required_bits = len(bin_msg)
        
        if required_bits > available_pixels:
            raise ValueError(
                f"Không đủ dung lượng! Cần {required_bits} bits "
                f"nhưng chỉ có {available_pixels} pixels tại peak={peak}. "
                f"Tối đa có thể nhúng {available_pixels // 8} bytes. "
                f"Message hiện tại: {len(message)} ký tự = {required_bits} bits."
            )
        
        stego = image.copy().astype(np.int32)
        # Shift các pixel từ peak + 1 sang phải (an toàn vì peak < 255)
        stego[image > peak] += 1
        
        msg_idx = 0
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if image[r, c] == peak and msg_idx < len(bin_msg):
                    if bin_msg[msg_idx] == '1':
                        stego[r, c] = peak + 1
                    # Nếu bit là '0', giữ nguyên giá trị peak
                    msg_idx += 1
                    
                if msg_idx >= len(bin_msg):
                    break
            if msg_idx >= len(bin_msg):
                break
                
        return np.clip(stego, 0, 255).astype(np.uint8), peak

    @staticmethod
    def extract(stego, key, peak):
        bin_msg = ""
        
        # Duyệt qua ảnh theo thứ tự để trích xuất bit
        for r in range(stego.shape[0]):
            for c in range(stego.shape[1]):
                val = stego[r, c]
                if val == peak:
                    bin_msg += "0"
                elif val == peak + 1:
                    bin_msg += "1"
                else:
                    # Bỏ qua các pixel không phải peak hoặc peak+1
                    continue
                
                # Kiểm tra kết thúc sớm (3 bytes NULL = 24 bits 0)
                if len(bin_msg) % 8 == 0 and len(bin_msg) >= 24:
                    if "000000000000000000000000" in bin_msg[-48:]:
                        break
            
            # Break vòng lặp ngoài nếu đã tìm thấy kết thúc
            if len(bin_msg) % 8 == 0 and len(bin_msg) >= 24:
                if "000000000000000000000000" in bin_msg[-48:]:
                    break
        
        # Chuyển bit sang byte và decode UTF-8
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8:
                break
            byte_val = int(byte_str, 2)
            if byte_val == 0:  # Dừng lại khi gặp byte NULL
                break
            byte_list.append(byte_val)
        
        try:
            return bytes(byte_list).decode('utf-8')
        except Exception as e:
            return bytes(byte_list).decode('utf-8', errors='ignore')