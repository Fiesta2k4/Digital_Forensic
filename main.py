import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Import nội bộ
from core.lsb_sub import LSB_Sub
from utils import metrics, security

class SteganoToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Steganography Analyzer - Group 4")
        self.root.geometry("1200x850")

        self.cover_path = tk.StringVar()
        self.key_k = tk.StringVar()
        self.text_file_path = tk.StringVar(value="Chưa chọn file")
        self.input_dir = "data/input"
        
        if not os.path.exists("data/output"): os.makedirs("data/output")
        self.setup_ui()

    def setup_ui(self):
        # LEFT PANEL
        left = ttk.LabelFrame(self.root, text=" Cấu hình ", padding=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Label(left, text="1. Chọn nguồn ảnh:").pack(anchor="w")
        self.folder_cb = ttk.Combobox(left, values=["BOSSbase_256", "SUNI_02", "SUNI_04"])
        self.folder_cb.pack(fill="x", pady=5)
        self.folder_cb.set("BOSSbase_256")
        ttk.Button(left, text="Duyệt ảnh", command=self.load_image).pack(fill="x")
        ttk.Label(left, textvariable=self.cover_path, font=("Arial", 7), wraplength=180).pack()

        ttk.Label(left, text="2. Khóa K:").pack(anchor="w", pady=(10,0))
        ttk.Entry(left, textvariable=self.key_k, show="*").pack(fill="x")

        ttk.Label(left, text="3. Tin nhắn:").pack(anchor="w", pady=(10,0))
        ttk.Button(left, text="Chọn file .txt", command=self.load_text).pack(fill="x")
        ttk.Label(left, textvariable=self.text_file_path, font=("Arial", 7)).pack()
        self.msg_input = tk.Text(left, height=4, width=25)
        self.msg_input.pack()

        ttk.Button(left, text="EMBED & ANALYZE", command=self.run_embed).pack(fill="x", pady=20)
        ttk.Button(left, text="EXTRACT", command=self.run_extract).pack(fill="x")
        ttk.Button(left, text="XEM BIỂU ĐỒ ROC (DEMO)", command=security.plot_roc_demo).pack(fill="x", pady=5)

        # RIGHT PANEL
        right = tk.Frame(self.root)
        right.pack(side="right", expand=True, fill="both", padx=10)

        img_f = tk.Frame(right)
        img_f.pack(fill="both", expand=True)
        self.l_cover = tk.Label(img_f, text="Cover", relief="solid")
        self.l_cover.pack(side="left", expand=True, padx=5)
        self.l_stego = tk.Label(img_f, text="Stego", relief="solid")
        self.l_stego.pack(side="right", expand=True, padx=5)

        # METRICS DISPLAY
        self.res_txt = tk.Text(right, height=10, font=("Consolas", 10), bg="#f0f0f0")
        self.res_txt.pack(fill="x", pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(initialdir=os.path.join(self.input_dir, self.folder_cb.get()))
        if path:
            self.cover_path.set(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.show_img(img, "c")

    def load_text(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if path: self.text_file_path.set(path)

    def show_img(self, img, t="c"):
        img_p = Image.fromarray(img).resize((380, 380))
        img_t = ImageTk.PhotoImage(img_p)
        if t == "c": self.l_cover.config(image=img_t); self.l_cover.image = img_t
        else: self.l_stego.config(image=img_t); self.l_stego.image = img_t

    def run_embed(self):
        if not self.cover_path.get() or not self.key_k.get(): return
        
        cover = cv2.imread(self.cover_path.get(), cv2.IMREAD_GRAYSCALE)
        msg = self.msg_input.get("1.0", tk.END).strip()
        if self.text_file_path.get() != "Chưa chọn file":
            with open(self.text_file_path.get(), 'r') as f: msg = f.read()

        try:
            t1 = time.time()
            stego = LSB_Sub.embed(cover, msg, self.key_k.get())
            t2 = time.time()

            # Tính Metrics
            aec = metrics.calculate_aec(msg, cover.shape)
            psnr = metrics.calculate_psnr(cover, stego)
            ssim = metrics.calculate_ssim(cover, stego)
            uiqi = metrics.calculate_uiqi(cover, stego)
            ncc = metrics.calculate_ncc(cover, stego)
            kl = security.get_kl_divergence(cover, stego) # Gọi đúng hàm tính KL
            rm, sm = security.rs_analysis_demo(stego)

            self.show_img(stego, "s")
            self.current_stego = stego

            # Hiển thị kết quả lên UI
            self.res_txt.delete("1.0", tk.END)
            res = f"[QUALITY METRICS]\n"
            res += f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}\n"
            res += f"AEC : {aec:.4f} bpp | UIQI: {uiqi:.4f} | NCC: {ncc:.4f}\n"
            res += f"Time: {(t2-t1)*1000:.2f} ms\n\n"
            res += f"[SECURITY ANALYSIS]\n"
            res += f"KL Divergence: {kl:.8f}\n"
            res += f"RS Analysis  : Rm={rm:.2f}, Sm={sm:.2f}\n"
            self.res_txt.insert(tk.END, res)

            # Hiện PDH
            h_c = security.get_pdh(cover)
            h_s = security.get_pdh(stego)
            plt.figure("PDH Analysis", figsize=(8,4))
            plt.plot(h_c[:30], 'b-', label='Cover')
            plt.plot(h_s[:30], 'r--', label='Stego')
            plt.legend(); plt.show()

        except Exception as e: messagebox.showerror("Error", str(e))

    def run_extract(self):
        if not hasattr(self, 'current_stego'): return
        msg = LSB_Sub.extract(self.current_stego, self.key_k.get())
        messagebox.showinfo("Extracted", msg)

if __name__ == "__main__":
    root = tk.Tk(); app = SteganoToolApp(root); root.mainloop()