import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import traceback
from PIL import Image, ImageTk
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Import các thuật toán từ thư mục core
from core.lsb_sub import LSB_Sub
from core.lsb_matching import LSB_Matching
from core.pvd import PVD
from core.emd import EMD
from core.histogram_shifting import HistogramShifting
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

# Import các công cụ đánh giá
from utils import metrics, security

# Khóa vị trí folder data dựa trên file main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class SteganoToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Steganography Analyzer - Group 4")
        self.root.geometry("1400x900")

        # --- KHAI BÁO CÁC BIẾN ---
        # Embed variables
        self.cover_path = tk.StringVar()
        self.key_k_embed = tk.StringVar()
        self.method_var = tk.StringVar(value="LSB Substitution")
        self.text_file_path = tk.StringVar(value="Chưa chọn file")
        
        # Extract variables
        self.stego_path = tk.StringVar()
        self.key_k_extract = tk.StringVar()
        self.method_extract_var = tk.StringVar(value="LSB Substitution")
        
        self.input_dir = os.path.join(BASE_DIR, "data", "input")
        self.output_dir = os.path.join(BASE_DIR, "data", "output")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.setup_ui()

    def setup_ui(self):
        # Tạo Notebook (Tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # TAB 1: EMBED
        tab_embed = ttk.Frame(notebook)
        notebook.add(tab_embed, text="📥 EMBED (Nhúng tin)")
        self.setup_embed_tab(tab_embed)
        
        # TAB 2: EXTRACT
        tab_extract = ttk.Frame(notebook)
        notebook.add(tab_extract, text="📤 EXTRACT (Trích xuất)")
        self.setup_extract_tab(tab_extract)
    
    def setup_embed_tab(self, parent):
        # PANEL TRÁI: ĐIỀU KHIỂN
        left = ttk.LabelFrame(parent, text=" Cấu hình Embed ", padding=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # 1. Chọn ảnh
        ttk.Label(left, text="1. Chọn ảnh cover:").pack(anchor="w")
        self.folder_cb = ttk.Combobox(left, values=["standard", "BOSSbase_256", "SUNI_02", "SUNI_04"])
        self.folder_cb.pack(fill="x", pady=5)
        self.folder_cb.set("standard")
        ttk.Button(left, text="Duyệt ảnh (.pgm)", command=self.load_image).pack(fill="x")
        ttk.Label(left, textvariable=self.cover_path, font=("Arial", 7), wraplength=180, foreground="blue").pack()

        # 2. Chọn thuật toán
        ttk.Label(left, text="2. Chọn thuật toán:").pack(anchor="w", pady=(10,0))
        methods = ["LSB Substitution", "LSB Matching", "PVD", "EMD", "Histogram Shifting", "Interpolation", "Difference Expansion"]
        method_menu = ttk.OptionMenu(left, self.method_var, methods[0], *methods)
        method_menu.pack(fill="x", pady=5)

        # 3. Khóa K
        ttk.Label(left, text="3. Nhập khóa bảo mật K:").pack(anchor="w", pady=(10,0))
        ttk.Entry(left, textvariable=self.key_k_embed, show="*").pack(fill="x")

        # 4. Tin nhắn
        ttk.Label(left, text="4. Tin nhắn bí mật:").pack(anchor="w", pady=(10,0))
        
        file_btn_frame = ttk.Frame(left)
        file_btn_frame.pack(fill="x")
        ttk.Button(file_btn_frame, text="Chọn file .txt", command=self.load_text).pack(side="left", expand=True)
        ttk.Button(file_btn_frame, text="Xóa file", command=self.clear_text_file).pack(side="left", expand=True)
        
        ttk.Label(left, textvariable=self.text_file_path, font=("Arial", 7), foreground="green").pack()
        
        ttk.Label(left, text="Hoặc nhập tay:").pack(anchor="w")
        self.msg_input = tk.Text(left, height=4, width=25)
        self.msg_input.pack()

        # 5. Nút thực thi
        ttk.Button(left, text="🔒 EMBED & ANALYZE", command=self.run_embed).pack(fill="x", pady=(20, 5))
        ttk.Button(left, text="💾 LƯU ẢNH STEGO", command=self.save_stego_image).pack(fill="x", pady=5)

        # PANEL PHẢI: KẾT QUẢ
        right = tk.Frame(parent)
        right.pack(side="right", expand=True, fill="both", padx=10)

        # Hiển thị ảnh
        img_f = tk.Frame(right)
        img_f.pack(fill="both", expand=True)
        self.l_cover = tk.Label(img_f, text="Ảnh Cover", relief="solid", borderwidth=1, bg="lightgray")
        self.l_cover.pack(side="left", expand=True, padx=5)
        self.l_stego = tk.Label(img_f, text="Ảnh Stego", relief="solid", borderwidth=1, bg="lightgray")
        self.l_stego.pack(side="right", expand=True, padx=5)

        # Khu vực hiện Metrics
        self.res_txt = tk.Text(right, height=12, font=("Consolas", 9), bg="#f8f8f8")
        self.res_txt.pack(fill="x", pady=10)
    
    def setup_extract_tab(self, parent):
        # PANEL TRÁI: ĐIỀU KHIỂN
        left = ttk.LabelFrame(parent, text=" Cấu hình Extract ", padding=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # 1. Chọn ảnh stego
        ttk.Label(left, text="1. Chọn ảnh stego:").pack(anchor="w")
        ttk.Button(left, text="Duyệt ảnh stego", command=self.load_stego_image).pack(fill="x")
        ttk.Label(left, textvariable=self.stego_path, font=("Arial", 7), wraplength=180, foreground="blue").pack()

        # 2. Chọn thuật toán
        ttk.Label(left, text="2. Chọn thuật toán:").pack(anchor="w", pady=(10,0))
        methods = ["LSB Substitution", "LSB Matching", "PVD", "EMD", "Histogram Shifting", "Interpolation", "Difference Expansion"]
        method_menu_extract = ttk.OptionMenu(left, self.method_extract_var, methods[0], *methods)
        method_menu_extract.pack(fill="x", pady=5)

        # 3. Khóa K
        ttk.Label(left, text="3. Nhập khóa K (phải đúng):").pack(anchor="w", pady=(10,0))
        ttk.Entry(left, textvariable=self.key_k_extract, show="*").pack(fill="x")
        
        ttk.Label(left, text="⚠️ Nếu nhập sai khóa K,\ntin nhắn sẽ bị sai!", 
                  foreground="red", font=("Arial", 8, "italic")).pack(anchor="w", pady=5)

        # 4. Nút thực thi
        ttk.Button(left, text="🔓 EXTRACT MESSAGE", command=self.run_extract_tab).pack(fill="x", pady=20)

        # PANEL PHẢI: KẾT QUẢ
        right = tk.Frame(parent)
        right.pack(side="right", expand=True, fill="both", padx=10)

        # Hiển thị ảnh
        img_f = tk.Frame(right)
        img_f.pack(fill="both", expand=True)
        self.l_stego_extract = tk.Label(img_f, text="Ảnh Stego", relief="solid", borderwidth=1, bg="lightgray")
        self.l_stego_extract.pack(expand=True, padx=5, pady=5)

        # Khu vực hiện tin nhắn
        ttk.Label(right, text="Tin nhắn trích xuất:").pack(anchor="w")
        self.extract_txt = tk.Text(right, height=15, font=("Arial", 11), bg="#ffffcc")
        self.extract_txt.pack(fill="both", expand=True, pady=5)

    def clear_text_file(self):
        """Xóa file text đã chọn"""
        self.text_file_path.set("Chưa chọn file")
        messagebox.showinfo("Thông báo", "Đã xóa file text. Hệ thống sẽ dùng tin nhắn nhập tay.")

    def load_image(self):
        sub_folder = self.folder_cb.get()
        initial_dir = os.path.join(self.input_dir, sub_folder)
        path = filedialog.askopenfilename(initialdir="data/input/", filetypes=[("PGM", "*.pgm")])
        if path:
            self.cover_path.set(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.show_img(img, "c")

    def load_text(self):
        path = filedialog.askopenfilename(initialdir="test/", filetypes=[("Văn bản", "*.txt")])
        if path:
            self.text_file_path.set(path)

    def show_img(self, img, t="c"):
        img_p = Image.fromarray(img).resize((380, 380))
        img_t = ImageTk.PhotoImage(img_p)
        if t == "c":
            self.l_cover.config(image=img_t, text="")
            self.l_cover.image = img_t
        else:
            self.l_stego.config(image=img_t, text="")
            self.l_stego.image = img_t

    def call_algorithm_embed(self, method_name, cover, msg, key):
        """
        Hàm trung gian để gọi các thuật toán khác nhau.
        Trả về: (ảnh_stego, tham_số_nhúng)
        """
        if method_name == "LSB Substitution":
            # Trả về: (stego_img, n_bits)
            return LSB_Sub.embed(cover, msg, key)
            
        elif method_name == "LSB Matching":
            # Trả về: (stego_img, n_bits)
            return LSB_Matching.embed(cover, msg, key)
            
        elif method_name == "PVD":
            # Trả về: (stego_img, "Adaptive")
            return PVD.embed(cover, msg, key)
            
        elif method_name == "EMD":
            # EMD trả về (stego_img, n_digits)
            return EMD.embed(cover, msg, key)
            
        elif method_name == "Histogram Shifting":
            # Trả về (stego_img, peak) - peak cần cho extract
            return HistogramShifting.embed(cover, msg, key)
            
        elif method_name == "Difference Expansion":
            # Trả về: (stego_img, layers)
            return DifferenceExpansion.embed(cover, msg, key)
            
        elif method_name == "Interpolation":
            # Trả về (stego_img, 1)
            return Interpolation.embed(cover, msg, key)
            
        else:
            raise ValueError(f"Thuật toán {method_name} chưa được tích hợp!")

    def save_stego_image(self):
        """Lưu ảnh stego vào thư mục output"""
        if not hasattr(self, 'current_stego'):
            messagebox.showwarning("Lưu ảnh", "Chưa có ảnh stego! Vui lòng EMBED trước.")
            return
        
        # Tạo tên file rõ ràng
        method = self.method_var.get().replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cover_name = os.path.basename(self.cover_path.get()).replace(".pgm", "")
        
        # Thêm n_digits vào filename (nếu có)
        param_suffix = ""
        if hasattr(self, 'embed_param') and isinstance(self.embed_param, int):
            param_suffix = f"_nd{self.embed_param}"
        
        filename = f"stego_{method}_{cover_name}{param_suffix}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # Lưu ảnh
        cv2.imwrite(filepath, self.current_stego)
        
        messagebox.showinfo("Lưu thành công", 
                           f"Đã lưu ảnh stego:\n{filename}\n\nVào thư mục: data/output/")
    
    def load_stego_image(self):
        """Load ảnh stego để extract"""
        path = filedialog.askopenfilename(
            initialdir=self.output_dir,
            title="Chọn ảnh stego",
            filetypes=[("All Images", "*.pgm *.png *.jpg *.bmp"), ("PGM", "*.pgm"), ("PNG", "*.png")]
        )
        if path:
            self.stego_path.set(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_p = Image.fromarray(img).resize((380, 380))
                img_t = ImageTk.PhotoImage(img_p)
                self.l_stego_extract.config(image=img_t, text="")
                self.l_stego_extract.image = img_t
                # Lưu để extract
                self.loaded_stego = img

    def run_embed(self):
        if not self.cover_path.get() or not self.key_k_embed.get():
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh và nhập khóa K!")
            return
        
        cover = cv2.imread(self.cover_path.get(), cv2.IMREAD_GRAYSCALE)
        key = self.key_k_embed.get()
        method = self.method_var.get()
        
        # LẤY TIN NHẮN VÀ LÀM SẠCH (TRÁNH LỖI FILE TEXT)
        if self.text_file_path.get() != "Chưa chọn file":
            try:
                # Dùng encoding='utf-8' để đọc được tiếng Việt/ký tự đặc biệt
                with open(self.text_file_path.get(), 'r', encoding='utf-8') as f:
                    msg = f.read().strip() # .strip() để bỏ dấu xuống dòng thừa ở cuối file
            except Exception as e:
                messagebox.showerror("Lỗi đọc file", f"Không thể đọc file text: {e}")
                return
        else:
            msg = self.msg_input.get("1.0", tk.END).strip()

        if not msg:
            messagebox.showwarning("Lỗi", "Tin nhắn trống!")
            return

        try:

            t1 = time.time()
            
            # GỌI HÀM TRUNG GIAN VỪA TẠO
            self.current_stego, self.embed_param = self.call_algorithm_embed(method, cover, msg, key)
            
            t2 = time.time()
            

            # Tính toán đánh giá
            aec = metrics.calculate_aec(msg, cover.shape)
            psnr = metrics.calculate_psnr(cover, self.current_stego)
            ssim = metrics.calculate_ssim(cover, self.current_stego)
            uiqi = metrics.calculate_uiqi(cover, self.current_stego)
            ncc = metrics.calculate_ncc(cover, self.current_stego)
            kl = security.get_kl_divergence(cover, self.current_stego)
            rm, sm = security.rs_analysis_demo(self.current_stego)

            self.show_img(self.current_stego, "s")
            self.current_stego = self.current_stego
            
            # Hiển thị Metrics lên Textbox
            self.res_txt.delete("1.0", tk.END)
            res = f"--- KẾT QUẢ PHÂN TÍCH ({method}) ---\n"
            
            # Sửa np.info thành thông tin thực tế từ self.embed_param
            if isinstance(self.embed_param, int):
                res += f"Mức độ nhúng (Payload): {self.embed_param} bit(s)/pixel\n"
            else:
                res += f"Tham số nhúng (Param): {self.embed_param}\n"
                
            res += f"Dung lượng tin nhắn: {len(msg)*8:,} bits\n"
            res += f"Dung lượng (AEC): {aec:.4f} bpp\n"
            res += f"----------------------------------\n"
            res += f"CHẤT LƯỢNG (Quality Metrics):\n"
            res += f"Chỉ số PSNR: {psnr:.2f} dB\n"
            res += f"Độ tương đồng SSIM: {ssim:.4f}\n"
            res += f"Chỉ số UIQI: {uiqi:.4f}\n"
            res += f"Hệ số NCC: {ncc:.4f}\n"
            res += f"Thời gian thực hiện: {(t2 - t1)*1000:.2f} ms\n"
            res += f"----------------------------------\n"
            res += f"BẢO MẬT (Security Analysis):\n"
            res += f"Độ chệch KL: {kl:.8f}\n"
            res += f"RS Analysis: Rm = {rm:.4f}, Sm = {sm:.4f}\n"
            
            self.res_txt.insert(tk.END, res)

            # Vẽ biểu đồ PDH
            h_c = security.get_pdh(cover)
            h_s = security.get_pdh(self.current_stego)
            plt.figure("Phân tích PDH", figsize=(8,4))
            plt.plot(h_c[:40], 'b-', label='Cover Image')
            plt.plot(h_s[:40], 'r--', label='Stego Image')
            plt.title(f"So sánh biểu đồ PDH - {method}")
            plt.legend()
            plt.show()

        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Lỗi thuật toán", str(e))

    def run_extract_tab(self):
        """Extract message từ tab Extract với key riêng"""
        if not self.stego_path.get() or not self.key_k_extract.get():
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh stego và nhập khóa K!")
            return
        
        if not hasattr(self, 'loaded_stego'):
            messagebox.showwarning("Lỗi", "Vui lòng load lại ảnh stego!")
            return
        
        method = self.method_extract_var.get()
        key = self.key_k_extract.get()
        
        # Lấy param từ tên file nếu có, hoặc dùng default
        filename = os.path.basename(self.stego_path.get())
        
        # Parse param từ metadata nếu có (hoặc dùng giá trị mặc định)
        # Với LSB thường là 1-4 bits, Histogram là peak value
        # Tạm thời dùng giá trị mặc định cho demo
        default_params = {
            "LSB Substitution": 1,
            "LSB Matching": 1,
            "PVD": None,
            "EMD": 1,
            "Histogram Shifting": None,  # Cần peak value
            "Interpolation": 1,
            "Difference Expansion": None
        }
        
        self.extract_txt.delete("1.0", tk.END)
        self.extract_txt.insert(tk.END, "Đang trích xuất...\n")
        self.root.update()
        
        try:
            msg = None
            
            if method == "LSB Substitution":
                # Thử với các n_bits khác nhau
                for n_bits in [1, 2, 3, 4]:
                    try:
                        temp_msg = LSB_Sub.extract(self.loaded_stego, key, n_bits=n_bits)
                        if temp_msg and len(temp_msg) > 0:
                            msg = temp_msg
                            break
                    except:
                        continue
                        
            elif method == "LSB Matching":
                for n_bits in [1, 2, 3, 4]:
                    try:
                        temp_msg = LSB_Matching.extract(self.loaded_stego, key, n_bits=n_bits)
                        if temp_msg and len(temp_msg) > 0:
                            msg = temp_msg
                            break
                    except:
                        continue
                        
            elif method == "PVD":
                msg = PVD.extract(self.loaded_stego, key)
                
            elif method == "Histogram Shifting":
                # Cần tìm peak value - thử các giá trị phổ biến
                hist_data = np.histogram(self.loaded_stego, bins=256, range=(0, 256))[0]
                peak_candidates = np.argsort(hist_data)[::-1][:5]  # Top 5 peaks
                
                for peak in peak_candidates:
                    if 0 < peak < 255:
                        try:
                            temp_msg = HistogramShifting.extract(self.loaded_stego, key, peak=int(peak))
                            if temp_msg and len(temp_msg) > 5:
                                msg = temp_msg
                                break
                        except:
                            continue
                            
            elif method == "EMD":
                # Cố gắng lấy n_digits từ tên file
                filename = os.path.basename(self.stego_path.get())
                n_digits_from_file = None
                
                # Parse "_nd{number}_" từ filename, vd: stego_EMD_1_nd256_20250112.png
                import re
                match = re.search(r'_nd(\d+)_', filename)
                if match:
                    n_digits_from_file = int(match.group(1))
                
                if n_digits_from_file:
                    # Nếu tìm thấy n_digits trong filename, dùng nó trực tiếp
                    try:
                        msg = EMD.extract(self.loaded_stego, key, n_digits=n_digits_from_file)
                    except Exception as e:
                        msg = None
                        self.extract_txt.delete("1.0", tk.END)
                        self.extract_txt.insert(tk.END, f"❌ LỖI EXTRACT EMD!\n\n{str(e)}\n\n")
                        self.extract_txt.insert(tk.END, f"n_digits = {n_digits_from_file}")
                        return
                else:
                    # Nếu không tìm thấy trong filename, báo lỗi
                    self.extract_txt.delete("1.0", tk.END)
                    self.extract_txt.insert(tk.END, "❌ KHÔNG TÌM THẤY n_digits!\n\n")
                    self.extract_txt.insert(tk.END, "Để extract EMD, ảnh phải được lưu bằng nút 'LƯU ẢNH STEGO'\n\n")
                    self.extract_txt.insert(tk.END, "Tên file phải có dạng:\nstego_EMD_..._nd{NUMBER}_TIMESTAMP.png\n\n")
                    self.extract_txt.insert(tk.END, "Ví dụ: stego_EMD_1_nd45_20250112_150000.png")
                    return
                
            elif method == "Difference Expansion":
                msg = DifferenceExpansion.extract(self.loaded_stego, key)
                
            elif method == "Interpolation":
                for n_bits in [1, 2, 3, 4]:
                    try:
                        temp_msg = Interpolation.extract(self.loaded_stego, key, n_bits=n_bits)
                        if temp_msg and len(temp_msg) > 0:
                            msg = temp_msg
                            break
                    except:
                        continue
            else:
                msg = "Thuật toán chưa được hỗ trợ."
            
            # Hiển thị kết quả
            self.extract_txt.delete("1.0", tk.END)
            
            if msg and len(msg) > 0:
                self.extract_txt.insert(tk.END, "✅ TRÍCH XUẤT THÀNH CÔNG!\n\n")
                self.extract_txt.insert(tk.END, "=" * 60 + "\n")
                self.extract_txt.insert(tk.END, msg)
                self.extract_txt.insert(tk.END, "\n" + "=" * 60 + "\n")
                self.extract_txt.insert(tk.END, f"\nĐộ dài: {len(msg)} ký tự")
            else:
                self.extract_txt.insert(tk.END, "❌ KHÔNG TÌM THẤY TIN NHẮN!\n\n")
                self.extract_txt.insert(tk.END, "Có thể do:\n")
                self.extract_txt.insert(tk.END, "• Sai khóa K\n")
                self.extract_txt.insert(tk.END, "• Sai thuật toán\n")
                self.extract_txt.insert(tk.END, "• Ảnh không phải stego image\n")
                
        except Exception as e:
            self.extract_txt.delete("1.0", tk.END)
            self.extract_txt.insert(tk.END, f"❌ LỖI TRÍCH XUẤT!\n\n{str(e)}\n\n")
            self.extract_txt.insert(tk.END, "Vui lòng kiểm tra lại:\n")
            self.extract_txt.insert(tk.END, "• Khóa K có đúng không?\n")
            self.extract_txt.insert(tk.END, "• Thuật toán có đúng không?")

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoToolApp(root)
    root.mainloop()