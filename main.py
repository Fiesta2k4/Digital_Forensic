import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        self.root.geometry("1200x850")

        # --- KHAI BÁO CÁC BIẾN (Sửa lỗi Attribute Error) ---
        self.cover_path = tk.StringVar()
        self.key_k = tk.StringVar()
        self.method_var = tk.StringVar(value="LSB Substitution") # BIẾN CHỌN THUẬT TOÁN
        self.text_file_path = tk.StringVar(value="Chưa chọn file")
        
        self.input_dir = os.path.join(BASE_DIR, "data", "input")
        self.output_dir = os.path.join(BASE_DIR, "data", "output")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.setup_ui()

    def setup_ui(self):
        # PANEL TRÁI: ĐIỀU KHIỂN
        left = ttk.LabelFrame(self.root, text=" Cấu hình hệ thống ", padding=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # 1. Chọn ảnh
        ttk.Label(left, text="1. Chọn nguồn ảnh:").pack(anchor="w")
        self.folder_cb = ttk.Combobox(left, values=["BOSSbase_256", "SUNI_02", "SUNI_04"])
        self.folder_cb.pack(fill="x", pady=5)
        self.folder_cb.set("BOSSbase_256")
        ttk.Button(left, text="Duyệt ảnh (.pgm)", command=self.load_image).pack(fill="x")
        ttk.Label(left, textvariable=self.cover_path, font=("Arial", 7), wraplength=180, foreground="blue").pack()

        # 2. Chọn phương pháp (6 phương pháp)
        ttk.Label(left, text="2. Chọn thuật toán:").pack(anchor="w", pady=(10,0))
        methods = ["LSB Substitution", "LSB Matching", "PVD", "EMD", "Histogram Shifting", "Interpolation", "Difference Expansion"]
        method_menu = ttk.OptionMenu(left, self.method_var, methods[0], *methods)
        method_menu.pack(fill="x", pady=5)

        # 3. Khóa K
        ttk.Label(left, text="3. Nhập khóa bảo mật K:").pack(anchor="w", pady=(10,0))
        ttk.Entry(left, textvariable=self.key_k, show="*").pack(fill="x")

        # 4. Tin nhắn & File Text
        ttk.Label(left, text="4. Tin nhắn bí mật:").pack(anchor="w", pady=(10,0))
        
        # Frame cho nút File
        file_btn_frame = ttk.Frame(left)
        file_btn_frame.pack(fill="x")
        ttk.Button(file_btn_frame, text="Chọn file .txt", command=self.load_text).pack(side="left", expand=True)
        # NÚT XÓA FILE THEO YÊU CẦU
        ttk.Button(file_btn_frame, text="Xóa file", command=self.clear_text_file).pack(side="left", expand=True)
        
        ttk.Label(left, textvariable=self.text_file_path, font=("Arial", 7), foreground="green").pack()
        
        ttk.Label(left, text="Hoặc nhập tay:").pack(anchor="w")
        self.msg_input = tk.Text(left, height=4, width=25)
        self.msg_input.pack()

        # 5. Nút thực thi
        ttk.Button(left, text="EMBED & ANALYZE", command=self.run_embed).pack(fill="x", pady=20)
        ttk.Button(left, text="EXTRACT", command=self.run_extract).pack(fill="x")

        # PANEL PHẢI: KẾT QUẢ
        right = tk.Frame(self.root)
        right.pack(side="right", expand=True, fill="both", padx=10)

        # Hiển thị ảnh
        img_f = tk.Frame(right)
        img_f.pack(fill="both", expand=True)
        self.l_cover = tk.Label(img_f, text="Ảnh Cover", relief="solid", borderwidth=1)
        self.l_cover.pack(side="left", expand=True, padx=5)
        self.l_stego = tk.Label(img_f, text="Ảnh Stego", relief="solid", borderwidth=1)
        self.l_stego.pack(side="right", expand=True, padx=5)

        # Khu vực hiện Metrics
        self.res_txt = tk.Text(right, height=12, font=("Consolas", 10), bg="#f8f8f8")
        self.res_txt.pack(fill="x", pady=10)

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
            # EMD chỉ nhúng 1 lớp, trả về (stego_img, 1)
            return EMD.embed(cover, msg, key), 1
            
        elif method_name == "Histogram Shifting":
            # Trả về (stego_img, 1)
            return HistogramShifting.embed(cover, msg, key), 1
            
        elif method_name == "Difference Expansion":
            # Trả về: (stego_img, layers)
            return DifferenceExpansion.embed(cover, msg, key)
            
        elif method_name == "Interpolation":
            # Trả về (stego_img, 1)
            return Interpolation.embed(cover, msg, key), 1
            
        else:
            raise ValueError(f"Thuật toán {method_name} chưa được tích hợp!")

    def run_embed(self):
        if not self.cover_path.get() or not self.key_k.get():
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng chọn ảnh và nhập khóa K!")
            return
        
        cover = cv2.imread(self.cover_path.get(), cv2.IMREAD_GRAYSCALE)
        key = self.key_k.get()
        method = self.method_var.get()
        
        # Lấy tin nhắn
        if self.text_file_path.get() != "Chưa chọn file":
            with open(self.text_file_path.get(), 'r', encoding='utf-8') as f:
                msg = f.read()
        else:
            msg = self.msg_input.get("1.0", tk.END).strip()

        if not msg:
            messagebox.showwarning("Lỗi", "Vui lòng nhập tin nhắn!")
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
            # Sửa thời gian: t2 - t1 đơn vị là giây, nhân 1000 để ra ms
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
            plt.legend(); plt.show()

        except Exception as e:
            messagebox.showerror("Lỗi thuật toán", str(e))

    def run_extract(self):
        if not hasattr(self, 'current_stego'): return
        
        method = self.method_var.get()
        key = self.key_k.get()
        # Lấy tham số nhúng đã lưu từ bước Embed
        param = getattr(self, 'embed_param', 1) 

        try:
            if method == "LSB Substitution":
                msg = LSB_Sub.extract(self.current_stego, key, n_bits=param)
            elif method == "LSB Matching":
                msg = LSB_Matching.extract(self.current_stego, key, n_bits=param)
            elif method == "PVD":
                # PVD adaptive nên thường không cần truyền n_bits cố định
                msg = PVD.extract(self.current_stego, key)
            elif method == "Difference Expansion":
                msg = DifferenceExpansion.extract(self.current_stego, key)
            elif method == "Histogram Shifting":
                msg = HistogramShifting.extract(self.current_stego, key, peak=param)
            elif method == "EMD":
                msg = EMD.extract(self.current_stego, key, n_digits=param)
            elif method == "Interpolation":
                msg = Interpolation.extract(self.current_stego, key, msg_len=param)
            else:
                msg = "Chưa cấu hình trích xuất cho phương pháp này."
                
            if msg:
                # Hiển thị kết quả trong cửa sổ mới
                result_window = tk.Toplevel(self.root)
                result_window.title("Tin nhắn trích xuất được")
                result_window.geometry("500x400")
                txt = tk.Text(result_window, font=("Arial", 11))
                txt.pack(expand=True, fill="both", padx=10, pady=10)
                txt.insert(tk.END, msg)
                txt.config(state="disabled") # Không cho sửa
            else:
                messagebox.showwarning("Kết quả", "Không tìm thấy tin nhắn (hoặc sai khóa K)!")

            

        except Exception as e:
            messagebox.showerror("Lỗi", "Sai tham số hoặc khóa K!")

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoToolApp(root)
    root.mainloop()