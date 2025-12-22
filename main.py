import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import time
import numpy as np

# Giả định bạn sẽ import các thuật toán sau khi viết xong
# from core.lsb_sub import LSB_Sub
# from core.pvd import PVD
# ... (TV1, TV2, TV3 sẽ code vào đây)

class SteganoToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Domain Steganography Tool - Group 4")
        self.root.geometry("1200x800")

        # Biến lưu trữ dữ liệu
        self.cover_path = tk.StringVar()
        self.stego_path = tk.StringVar()
        self.key_k = tk.StringVar()
        self.method_var = tk.StringVar(value="LSB Substitution")
        self.input_dir = "data/input" # Thư mục gốc chứa 3 folder bạn nêu
        
        self.setup_ui()

    def setup_ui(self):
        # --- PANEL TRÁI: Cấu hình ---
        left_panel = ttk.LabelFrame(self.root, text=" Cấu hình nhúng ", padding=10)
        left_panel.pack(side="left", fill="y", padx=10, pady=10)

        # 1. Chọn Folder & Ảnh
        ttk.Label(left_panel, text="Chọn nguồn ảnh:").pack(anchor="w")
        self.folder_cb = ttk.Combobox(left_panel, values=["BOSSbase_256", "SUNI_02", "SUNI_04"])
        self.folder_cb.pack(fill="x", pady=5)
        self.folder_cb.set("BOSSbase_256")

        ttk.Button(left_panel, text="Chọn ảnh từ nguồn", command=self.load_image).pack(fill="x", pady=5)
        ttk.Label(left_panel, textvariable=self.cover_path, foreground="blue", wraplength=200).pack()

        # 2. Chọn Phương pháp (6 phương pháp)
        ttk.Label(left_panel, text="Phương pháp:").pack(anchor="w", pady=(10, 0))
        methods = [
            "LSB Substitution", "LSB Matching", 
            "PVD (Pixel Value Differencing)", "EMD (Exploiting Modification Direction)",
            "Histogram Shifting (Reversible)", "Difference Expansion (Reversible)"
        ]
        method_menu = ttk.OptionMenu(left_panel, self.method_var, methods[0], *methods)
        method_menu.pack(fill="x", pady=5)

        # 3. Khóa K & Tin nhắn
        ttk.Label(left_panel, text="Khóa bảo mật K:").pack(anchor="w")
        ttk.Entry(left_panel, textvariable=self.key_k, show="*").pack(fill="x", pady=5)

        ttk.Label(left_panel, text="Tin nhắn bí mật:").pack(anchor="w")
        self.msg_text = tk.Text(left_panel, height=5, width=25)
        self.msg_text.pack(pady=5)

        # 4. Nút bấm thực hiện
        ttk.Button(left_panel, text="BẮT ĐẦU NHÚNG (EMBED)", command=self.process_embed).pack(fill="x", pady=20)
        ttk.Button(left_panel, text="TRÍCH XUẤT (EXTRACT)", command=self.process_extract).pack(fill="x")
        
        # --- PANEL PHẢI: Hiển thị & Kết quả ---
        right_panel = tk.Frame(self.root)
        right_panel.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # Khu vực hiển thị ảnh
        img_frame = tk.Frame(right_panel)
        img_frame.pack(fill="both", expand=True)

        self.canvas_cover = tk.Label(img_frame, text="Ảnh Cover", borderwidth=2, relief="groove")
        self.canvas_cover.pack(side="left", padx=10, expand=True)

        self.canvas_stego = tk.Label(img_frame, text="Ảnh Stego", borderwidth=2, relief="groove")
        self.canvas_stego.pack(side="right", padx=10, expand=True)

        # Bảng kết quả (Chỉ số đánh giá)
        res_frame = ttk.LabelFrame(right_panel, text=" Chỉ số đánh giá (Metrics) ")
        res_frame.pack(fill="x", pady=10)

        self.psnr_label = ttk.Label(res_frame, text="PSNR: -- dB")
        self.psnr_label.grid(row=0, column=0, padx=20, pady=5)

        self.ssim_label = ttk.Label(res_frame, text="SSIM: --")
        self.ssim_label.grid(row=0, column=1, padx=20, pady=5)

        self.time_label = ttk.Label(res_frame, text="Thời gian: -- ms")
        self.time_label.grid(row=0, column=2, padx=20, pady=5)

        self.reversible_label = ttk.Label(res_frame, text="Khôi phục ảnh gốc: --")
        self.reversible_label.grid(row=0, column=3, padx=20, pady=5)

    def load_image(self):
        # Mở hộp thoại chọn file trong folder đã chọn
        sub_folder = self.folder_cb.get()
        initial_dir = os.path.join(self.input_dir, sub_folder)
        
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Chọn ảnh .pgm",
            filetypes=(("PGM files", "*.pgm"), ("All files", "*.*"))
        )
        
        if file_path:
            self.cover_path.set(file_path)
            # Hiển thị ảnh lên giao diện
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(img, "cover")

    def display_image(self, img, type="cover"):
        # Chuyển đổi từ OpenCV sang Tkinter
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_pil)

        if type == "cover":
            self.canvas_cover.config(image=img_tk, text="")
            self.canvas_cover.image = img_tk
        else:
            self.canvas_stego.config(image=img_tk, text="")
            self.canvas_stego.image = img_tk

    def process_embed(self):
        if not self.cover_path.get() or not self.key_k.get():
            messagebox.showwarning("Thiếu dữ liệu", "Vui lòng chọn ảnh và nhập khóa K!")
            return

        # 1. Lấy dữ liệu từ UI
        cover_img = cv2.imread(self.cover_path.get(), cv2.IMREAD_GRAYSCALE)
        message = self.msg_text.get("1.0", tk.END).strip()
        key = self.key_k.get()
        method = self.method_var.get()

        # 2. Mô phỏng xử lý (TV1, 2, 3 sẽ thay bằng hàm thật tại đây)
        start_time = time.time()
        
        # GIẢ LẬP: Ở đây bạn sẽ gọi core.lsb.embed(...)
        time.sleep(0.5) # Giả lập thời gian chạy
        stego_img = cover_img.copy() 
        # Thêm nhiễu nhẹ để giả lập có nhúng
        stego_img = cv2.add(stego_img, np.random.randint(0, 2, cover_img.shape, dtype=np.uint8))
        
        end_time = time.time()

        # 3. Hiển thị kết quả & Đánh giá (TV4 viết module này)
        self.display_image(stego_img, "stego")
        
        # Tính toán giả lập PSNR
        mse = np.mean((cover_img - stego_img) ** 2)
        psnr = 100 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        
        self.psnr_label.config(text=f"PSNR: {psnr:.2f} dB")
        self.time_label.config(text=f"Thời gian: {(end_time - start_time)*1000:.1f} ms")
        messagebox.showinfo("Thành công", f"Đã nhúng bằng phương pháp {method}")

    def process_extract(self):
        # TV1, 2, 3 sẽ viết code trích xuất dựa trên khóa K
        # Nếu khóa K sai, hiển thị message lỗi hoặc rác
        key = self.key_k.get()
        if key == "123": # Giả lập key đúng
            messagebox.showinfo("Kết quả trích xuất", "Tin nhắn: Hello world!")
        else:
            messagebox.showwarning("Extract rác", "Sai khóa K! Dữ liệu trích xuất: x@#$!128*&")

if __name__ == "__main__":
    root = tk.Tk()
    app = SteganoToolApp(root)
    root.mainloop()