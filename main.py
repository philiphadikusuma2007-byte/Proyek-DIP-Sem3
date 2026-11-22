# =============================================================================
# main.py - TAMPILAN PROGRAM "DETEKSI SISA OBAT DALAM BLISTER"
#
# Library yang dipakai di file ini HANYA untuk tampilan (UI):
#   tkinter  -> jendela, tombol, tab, kanvas (bawaan Python)
#   threading-> agar jendela tidak "macet" selama proses berjalan
# Gambar ditampilkan dengan format PPM yang disusun manual dari list piksel
# (tidak memakai PIL untuk tampilan). Semua pengolahan citra ada di
# citra.py (rumus) dan deteksi.py (alur).
#
# Cara menjalankan:  python main.py
# =============================================================================

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from citra import potong, kontur, dilasi
from deteksi import analisis

WARNA_LATAR = "#f4f5f7"
WARNA_PANEL = "#ffffff"
WARNA_AKSEN = "#1f6f5c"
WARNA_TEKS = "#1d2433"
WARNA_REDUP = "#6b7280"
WARNA_OBAT = (40, 220, 90)
WARNA_TOLAK = (235, 70, 60)
MAKS_TAMPIL = 640             # sisi terpanjang maksimum gambar di tab Hasil
MAKS_KECIL = 260              # sisi terpanjang gambar di tab Tahapan


# ======================= KONVERSI LIST PIKSEL -> GAMBAR TK ===================
def ke_photoimage(R, G, B, lebar, tinggi, maks_sisi):
    """List piksel -> tk.PhotoImage lewat format PPM (P6) yang disusun manual.
    Sekalian diperkecil/diperbesar (tetangga terdekat) agar pas di layar."""
    skala = maks_sisi / max(lebar, tinggi)
    lb = max(1, int(lebar * skala))
    tb = max(1, int(tinggi * skala))
    sumber_x = [min(lebar - 1, int(x / skala)) for x in range(lb)]
    data = bytearray(lb * tb * 3)
    j = 0
    for y in range(tb):
        awal = min(tinggi - 1, int(y / skala)) * lebar
        for sx in sumber_x:
            i = awal + sx
            data[j] = potong(R[i])
            data[j + 1] = potong(G[i])
            data[j + 2] = potong(B[i])
            j += 3
    ppm = b"P6 %d %d 255\n" % (lb, tb) + bytes(data)
    return tk.PhotoImage(data=ppm), skala


def abu_ke_rgb(data, skala_nilai=1.0):
    nilai = [v * skala_nilai for v in data]
    return nilai, nilai, nilai


def gambar_overlay(hasil):
    """Foto potongan + isi obat diwarnai hijau transparan + kontur tebal,
    region yang ditolak diberi kontur merah tipis."""
    w = hasil["lebar"]
    h = hasil["tinggi"]
    n = w * h
    R, G, B = [list(k) for k in hasil["asli"]]
    tolak = [0] * n
    for d in hasil["ditolak"]:
        for i in d["piksel"]:
            tolak[i] = 1
    garis = kontur(tolak, w, h)
    for i in range(n):
        if garis[i]:
            R[i], G[i], B[i] = WARNA_TOLAK
    obat = [0] * n
    for d in hasil["obat"]:
        for i in d["piksel"]:
            obat[i] = 1
            R[i] = 0.6 * R[i] + 0.4 * WARNA_OBAT[0]
            G[i] = 0.6 * G[i] + 0.4 * WARNA_OBAT[1]
            B[i] = 0.6 * B[i] + 0.4 * WARNA_OBAT[2]
    garis = dilasi(kontur(obat, w, h), w, h, 3)      # kontur ditebalkan
    for i in range(n):
        if garis[i]:
            R[i], G[i], B[i] = WARNA_OBAT
    return R, G, B


# ================================ APLIKASI ===================================
class Aplikasi:
    def __init__(self, root):
        self.root = root
        self.path = None
        self.hasil = None
        self.gambar_tk = []          # simpan referensi agar tidak dihapus GC
        root.title("Deteksi Sisa Obat dalam Blister - Pengolahan Citra Digital")
        root.geometry("1280x800")
        root.minsize(1000, 640)
        root.configure(bg=WARNA_LATAR)
        self.buat_gaya()
        self.buat_tata_letak()

    # ------------------------------------------------------------------ UI
    def buat_gaya(self):
        gaya = ttk.Style()
        try:
            gaya.theme_use("clam")
        except tk.TclError:
            pass
        gaya.configure("TNotebook", background=WARNA_LATAR, borderwidth=0)
        gaya.configure("TNotebook.Tab", padding=(14, 6), font=("Segoe UI", 10))
        gaya.map("TNotebook.Tab", background=[("selected", WARNA_PANEL)])
        gaya.configure("Treeview", rowheight=22, font=("Consolas", 9))
        gaya.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def buat_tata_letak(self):
        kiri = tk.Frame(self.root, bg=WARNA_PANEL, width=320, padx=18, pady=18)
        kiri.pack(side="left", fill="y")
        kiri.pack_propagate(False)

        tk.Label(kiri, text="Deteksi Sisa Obat", font=("Segoe UI", 16, "bold"),
                 bg=WARNA_PANEL, fg=WARNA_TEKS).pack(anchor="w")
        tk.Label(kiri, text="Pengolahan Citra Digital tanpa Machine Learning",
                 font=("Segoe UI", 9), bg=WARNA_PANEL, fg=WARNA_REDUP, wraplength=280,
                 justify="left").pack(anchor="w", pady=(0, 14))

        tombol = dict(font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", pady=8)
        tk.Button(kiri, text="Pilih Foto Blister...", bg="#e5e7eb", fg=WARNA_TEKS,
                  activebackground="#d1d5db", command=self.pilih_foto, **tombol).pack(fill="x")
        self.label_file = tk.Label(kiri, text="Belum ada foto", font=("Segoe UI", 9),
                                   bg=WARNA_PANEL, fg=WARNA_REDUP, wraplength=280, justify="left")
        self.label_file.pack(anchor="w", pady=(6, 10))
        self.tombol_deteksi = tk.Button(kiri, text="DETEKSI", bg=WARNA_AKSEN, fg="white",
                                        activebackground="#17594a", activeforeground="white",
                                        command=self.mulai_deteksi, state="disabled", **tombol)
        self.tombol_deteksi.pack(fill="x")
        self.tombol_simpan = tk.Button(kiri, text="Simpan Gambar Hasil...", bg="#e5e7eb", fg=WARNA_TEKS,
                                       activebackground="#d1d5db", command=self.simpan,
                                       state="disabled", **tombol)
        self.tombol_simpan.pack(fill="x", pady=(8, 0))

        kotak = tk.Frame(kiri, bg="#eef7f4", padx=14, pady=12)
        kotak.pack(fill="x", pady=(18, 10))
        tk.Label(kotak, text="SISA OBAT", font=("Segoe UI", 9, "bold"), bg="#eef7f4",
                 fg=WARNA_AKSEN).pack(anchor="w")
        self.label_jumlah = tk.Label(kotak, text="-", font=("Segoe UI", 44, "bold"),
                                     bg="#eef7f4", fg=WARNA_AKSEN)
        self.label_jumlah.pack(anchor="w")
        self.label_rincian = tk.Label(kotak, text="", font=("Segoe UI", 9), bg="#eef7f4",
                                      fg=WARNA_TEKS, justify="left", wraplength=260)
        self.label_rincian.pack(anchor="w")

        tk.Label(kiri, text="Perbaikan citra (dipilih otomatis)", font=("Segoe UI", 10, "bold"),
                 bg=WARNA_PANEL, fg=WARNA_TEKS).pack(anchor="w", pady=(6, 4))
        self.teks_langkah = tk.Text(kiri, height=14, font=("Consolas", 8), bg="#fafafa",
                                    relief="flat", wrap="word", fg=WARNA_TEKS)
        self.teks_langkah.pack(fill="both", expand=True)
        self.teks_langkah.configure(state="disabled")

        self.status = tk.Label(kiri, text="Siap.", font=("Segoe UI", 9), bg=WARNA_PANEL,
                               fg=WARNA_REDUP, wraplength=280, justify="left")
        self.status.pack(anchor="w", pady=(8, 0))
        self.progres = ttk.Progressbar(kiri, mode="indeterminate")
        self.progres.pack(fill="x", pady=(4, 0))

        kanan = tk.Frame(self.root, bg=WARNA_LATAR, padx=12, pady=12)
        kanan.pack(side="left", fill="both", expand=True)
        self.tab = ttk.Notebook(kanan)
        self.tab.pack(fill="both", expand=True)

        self.tab_hasil = tk.Frame(self.tab, bg=WARNA_PANEL)
        self.tab_tahap = tk.Frame(self.tab, bg=WARNA_PANEL)
        self.tab_region = tk.Frame(self.tab, bg=WARNA_PANEL)
        self.tab.add(self.tab_hasil, text="Hasil")
        self.tab.add(self.tab_tahap, text="Tahapan Pengolahan")
        self.tab.add(self.tab_region, text="Keputusan per Region")

        self.kanvas_hasil = tk.Canvas(self.tab_hasil, bg=WARNA_PANEL, highlightthickness=0)
        self.kanvas_hasil.pack(fill="both", expand=True)
        self.kanvas_hasil.create_text(20, 20, anchor="nw", fill=WARNA_REDUP, font=("Segoe UI", 11),
                                      text="Pilih foto blister lalu tekan DETEKSI.")

        # tab tahapan: kanvas yang bisa di-scroll
        self.kanvas_tahap = tk.Canvas(self.tab_tahap, bg=WARNA_PANEL, highlightthickness=0)
        gulir = ttk.Scrollbar(self.tab_tahap, orient="vertical", command=self.kanvas_tahap.yview)
        self.kanvas_tahap.configure(yscrollcommand=gulir.set)
        gulir.pack(side="right", fill="y")
        self.kanvas_tahap.pack(side="left", fill="both", expand=True)
        self.kanvas_tahap.bind_all("<MouseWheel>", self.gulir_mouse)

        kolom = ("no", "luas", "keliling", "kompak", "isi", "lonjong", "d_meja", "d_foil", "biru",
                 "terang", "biji", "keputusan")
        self.tabel = ttk.Treeview(self.tab_region, columns=kolom, show="headings")
        judul = {"no": "No", "luas": "Luas", "keliling": "Keliling", "kompak": "Kompak",
                 "isi": "Isi", "lonjong": "Lonjong", "d_meja": "dMeja", "d_foil": "dFoil",
                 "biru": "dBiru", "terang": "Terang", "biji": "Halus", "keputusan": "Keputusan"}
        for k in kolom:
            self.tabel.heading(k, text=judul[k])
            self.tabel.column(k, width=170 if k == "keputusan" else 66, minwidth=50,
                              anchor="w" if k == "keputusan" else "center", stretch=(k == "keputusan"))
        self.tabel.pack(fill="both", expand=True, padx=8, pady=8)
        tk.Label(self.tab_region, bg=WARNA_PANEL, fg=WARNA_REDUP, font=("Segoe UI", 9), justify="left",
                 wraplength=880,
                 text=("Luas dalam piksel. Keliling dari kode rantai 8-arah, kompak = 4*pi*luas/keliling^2, "
                       "isi = luas / luas elips momen, lonjong = sumbu panjang / sumbu pendek.  "
                       "dMeja / dFoil = jarak warna RGB ke meja / foil.  Terang = kecerahan / foil.  "
                       "Halus = bagian region yang halus & mirip obat (biji region growing).  "
                       "dBiru = b(foil) - b(region), b = B/(R+G+B).\n"
                       "Alasan ditolak: kosong (warna meja), di meja (cincin berwarna meja), kasar (tidak halus), "
                       "foil / plastik (warna sama dengan foil), pantulan (putih jenuh), bentuk, luar, serpihan.")
                 ).pack(anchor="w", padx=8, pady=(0, 8))

    def gulir_mouse(self, event):
        if self.tab.index("current") == 1:
            self.kanvas_tahap.yview_scroll(int(-event.delta / 120), "units")

    # ------------------------------------------------------------- AKSI
    def pilih_foto(self):
        path = filedialog.askopenfilename(
            title="Pilih foto blister obat",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp"), ("Semua file", "*.*")])
        if not path:
            return
        self.path = path
        self.label_file.config(text=os.path.basename(path))
        self.tombol_deteksi.config(state="normal")
        self.status.config(text="Foto dipilih. Tekan DETEKSI.")

    def mulai_deteksi(self):
        if not self.path:
            return
        self.tombol_deteksi.config(state="disabled")
        self.tombol_simpan.config(state="disabled")
        self.progres.start(12)
        threading.Thread(target=self.kerja, daemon=True).start()

    def kerja(self):
        try:
            hasil = analisis(self.path, progres=lambda t: self.root.after(0, self.status.config, {"text": t}))
            self.root.after(0, self.tampilkan, hasil)
        except Exception as e:                       # tampilkan error di UI
            self.root.after(0, self.gagal, e)

    def gagal(self, e):
        self.progres.stop()
        self.tombol_deteksi.config(state="normal")
        self.status.config(text="Gagal: " + str(e))
        messagebox.showerror("Gagal memproses", str(e))

    def simpan(self):
        if not self.gambar_tk:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            initialfile="hasil_deteksi.png")
        if path:
            self.gambar_hasil_penuh.write(path, format="png")
            self.status.config(text="Disimpan: " + path)

    # ------------------------------------------------------------ TAMPIL
    def tampilkan(self, hasil):
        self.hasil = hasil
        self.progres.stop()
        self.tombol_deteksi.config(state="normal")
        self.tombol_simpan.config(state="normal")
        self.gambar_tk = []

        utuh = hasil["jumlah"] - hasil["jumlah_potongan"]
        self.label_jumlah.config(text=str(hasil["jumlah"]))
        rincian = "%d butir utuh" % utuh
        if hasil["jumlah_potongan"]:
            rincian += " + %d potongan" % hasil["jumlah_potongan"]
        rincian += "\n%d region ditolak  |  %.1f detik" % (len(hasil["ditolak"]), hasil["waktu"])
        self.label_rincian.config(text=rincian)
        self.status.config(text="Selesai.")

        baris = []
        for nama, dipakai, alasan in hasil["langkah"]:
            baris.append(("[v] " if dipakai else "[ ] ") + nama + "\n    " + alasan)
        baris.append("")
        baris.append("Blister: " + hasil["sumber_blister"])
        baris.append("  eta Otsu Cb = %.2f" % hasil["eta_cb"])
        if hasil["warna_meja"]:
            baris.append("Warna meja (median) = %s" % (hasil["warna_meja"],))
        baris.append("Warna foil (median) = %s" % (hasil["warna_foil"],))
        self.teks_langkah.configure(state="normal")
        self.teks_langkah.delete("1.0", "end")
        self.teks_langkah.insert("1.0", "\n".join(baris))
        self.teks_langkah.configure(state="disabled")

        self.tampil_hasil(hasil)
        self.tampil_tahapan(hasil)
        self.tampil_region(hasil)

    def tampil_hasil(self, hasil):
        k = self.kanvas_hasil
        k.delete("all")
        w = hasil["lebar"]
        h = hasil["tinggi"]

        # kiri: foto utuh (diperkecil) + kotak blister
        Rk, Gk, Bk, wk, hk = hasil["kasar"]["kecil"]
        img_kecil, s1 = ke_photoimage(Rk, Gk, Bk, wk, hk, 300)
        self.gambar_tk.append(img_kecil)
        k.create_text(20, 14, anchor="nw", text="Foto input", font=("Segoe UI", 10, "bold"), fill=WARNA_TEKS)
        k.create_image(20, 38, anchor="nw", image=img_kecil)
        x0, y0, x1, y1 = hasil["kasar"]["kotak_kecil"]
        k.create_rectangle(20 + x0 * s1, 38 + y0 * s1, 20 + x1 * s1, 38 + y1 * s1, outline="#facc15", width=2)
        k.create_text(20, 38 + hk * s1 + 8, anchor="nw", fill=WARNA_REDUP, font=("Segoe UI", 9),
                      text="Kotak kuning = daerah blister\nyang dipotong dari foto asli")

        # kanan: hasil deteksi (ukuran menyesuaikan ruang kanvas yang tersedia)
        R, G, B = gambar_overlay(hasil)
        k.update_idletasks()
        ruang_x = max(k.winfo_width() - 350 - 20, 300)
        ruang_y = max(k.winfo_height() - 38 - 50, 300)
        sisi = min(MAKS_TAMPIL, ruang_x * max(w, h) / w, ruang_y * max(w, h) / h)
        img, s2 = ke_photoimage(R, G, B, w, h, sisi)
        self.gambar_tk.append(img)
        self.gambar_hasil_penuh = ke_photoimage(R, G, B, w, h, max(w, h))[0]
        ox = 350
        oy = 38
        k.create_text(ox, 14, anchor="nw", font=("Segoe UI", 10, "bold"), fill=WARNA_TEKS,
                      text="Hasil deteksi: %d obat tersisa" % hasil["jumlah"])
        k.create_image(ox, oy, anchor="nw", image=img)
        nomor = 0
        for d in sorted(hasil["obat"], key=lambda d: (round(d["cy"] / 25), d["cx"])):
            nomor += 1
            cx = ox + d["cx"] * s2
            cy = oy + d["cy"] * s2
            r = 13
            k.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#14532d", outline="white", width=2)
            k.create_text(cx, cy, text=str(nomor) + ("*" if d["potongan"] else ""),
                          fill="white", font=("Segoe UI", 10, "bold"))
        for d in hasil["ditolak"]:
            k.create_text(ox + d["cx"] * s2, oy + d["cy"] * s2, text=d["alasan"], fill="#fecaca",
                          font=("Segoe UI", 8, "bold"))
        k.create_text(ox, oy + h * s2 + 8, anchor="nw", fill=WARNA_REDUP, font=("Segoe UI", 9),
                      text="Hijau = obat (bernomor, * = potongan obat).  Merah = region yang ditolak + alasannya.")

    def tampil_tahapan(self, hasil):
        k = self.kanvas_tahap
        k.delete("all")
        w = hasil["lebar"]
        h = hasil["tinggi"]
        panel = [
            ("1. Potongan blister (asli)", hasil["asli"]),
            ("2. Setelah perbaikan citra", hasil["diperbaiki"]),
            ("3. Komponen Cb (YCbCr)", abu_ke_rgb([(v - 100) * 4 for v in hasil["cb"]])),
            ("4. Energi tekstur highpass", abu_ke_rgb(hasil["skor_foil"])),
            ("5. Mask foil (Otsu tekstur)", abu_ke_rgb(hasil["foil"], 255)),
            ("6. Mask blister + oktagon", self.gabung_mask(hasil["blister_asli"], hasil["blister"])),
            ("7. Std lokal (gelap = halus)", abu_ke_rgb(hasil["std"], 8)),
            ("8. Jarak warna ke foil lokal", abu_ke_rgb(hasil["jarak_foil"], 2)),
            ("9. Biji (halus & mirip obat)", abu_ke_rgb(hasil["biji"], 255)),
            ("10. Setelah region growing", abu_ke_rgb(hasil["calon"], 255)),
            ("11. Putih jenuh (pantulan)", abu_ke_rgb(hasil["kilau"], 255)),
            ("12. Hasil akhir", gambar_overlay(hasil)),
        ]
        lebar_kanvas = max(k.winfo_width(), 900)
        kolom = max(1, (lebar_kanvas - 20) // (MAKS_KECIL + 20))
        x = 20
        y = 16
        tinggi_baris = 0
        for j, (judul, (R, G, B)) in enumerate(panel):
            img, s = ke_photoimage(R, G, B, w, h, MAKS_KECIL)
            self.gambar_tk.append(img)
            if j % kolom == 0 and j > 0:
                x = 20
                y += tinggi_baris + 40
                tinggi_baris = 0
            k.create_text(x, y, anchor="nw", text=judul, font=("Segoe UI", 9, "bold"), fill=WARNA_TEKS)
            k.create_image(x, y + 20, anchor="nw", image=img)
            tinggi_baris = max(tinggi_baris, img.height())
            x += MAKS_KECIL + 20
        k.configure(scrollregion=(0, 0, lebar_kanvas, y + tinggi_baris + 60))

    def gabung_mask(self, asli, oktagon):
        # putih = mask blister, abu-abu = tambahan dari bungkus oktagon
        R = [255 if a else (110 if o else 0) for a, o in zip(asli, oktagon)]
        return R, R, R

    def tampil_region(self, hasil):
        self.tabel.delete(*self.tabel.get_children())
        semua = [(d, "OBAT" + (" (potongan)" if d["potongan"] else "")) for d in hasil["obat"]]
        semua += [(d, "ditolak: " + d["alasan"]) for d in hasil["ditolak"] if "isi" in d]
        semua.sort(key=lambda t: -t[0]["luas"])
        for no, (d, kep) in enumerate(semua, 1):
            self.tabel.insert("", "end", values=(
                no, d["luas"], "%.0f" % d.get("keliling", 0), "%.2f" % d.get("kompak", 0),
                "%.2f" % d["isi"], "%.2f" % d["lonjong"],
                "%.0f" % d.get("d_meja", 0), "%.0f" % d.get("d_foil", 0),
                "%+.3f" % d.get("selisih_biru", 0), "%.2f" % d.get("terang", 0),
                "%.2f" % d.get("biji", 0), kep))


if __name__ == "__main__":
    try:
        # Windows: tampilan tajam di layar dengan skala DPI > 100% (UI saja)
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    Aplikasi(root)
    root.mainloop()
