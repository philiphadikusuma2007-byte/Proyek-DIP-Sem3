# ============================================================================
# tampilan.py  -  JENDELA (GUI) untuk deteksi sisa obat pada blister
#
# Bagian ini HANYA tampilan. Library yang dipakai (diizinkan untuk UI):
#   tkinter (jendela), matplotlib (menggambar), Pillow (mengubah data piksel
#   menjadi gambar yang bisa ditampilkan).
# Semua pengolahan citra ada di deteksi.py + folder dip/ (tanpa library).
#
# Cara pakai (dari main.py):
#     from tampilan import jalankan_jendela
#     jalankan_jendela(fungsi_analisis)    # fungsi_analisis(path) -> dict hasil
# ============================================================================
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ------------------------------- WARNA TEMA ---------------------------------
BG_UTAMA = "#14171f"      # latar jendela
BG_PANEL = "#1c2029"      # latar sidebar / bar
BG_KARTU = "#232834"      # latar kartu penjelasan
BG_PILIH = "#2c3342"      # item sidebar yang dipilih
GARIS = "#323949"         # garis pemisah
TEKS = "#e6e9ef"          # teks utama
TEKS_REDUP = "#8b93a5"    # teks kedua
AKSEN = "#22c55e"         # hijau = obat
AKSEN_GELAP = "#16a34a"
MERAH = "#ef4444"
ABU = "#4b5263"           # badge "dilewati"

HURUF = "Segoe UI"
HURUF_MONO = "Consolas"

# warna badge per materi (dua angka pertama nama materi)
WARNA_MATERI = {"01": "#64748b", "02": "#0ea5e9", "03": "#8b5cf6", "04": "#f59e0b",
                "05": "#ec4899", "06": "#14b8a6", "07": "#a3a3a3", "08": "#3b82f6",
                "09": "#f97316"}


def warna_materi(materi):
    return WARNA_MATERI.get(materi[:2], "#06b6d4")


# --------------------- DATA PIKSEL -> GAMBAR (Pillow) -----------------------
def ke_gambar(g):
    # g = dict gambar dari deteksi.py: rgb (list R, G, B) atau satu kanal 0..255
    ukuran = (g["w"], g["h"])
    if g["jenis"] == "rgb":
        n = g["w"] * g["h"]
        data = bytearray(3 * n)
        data[0::3] = bytes(g["R"])
        data[1::3] = bytes(g["G"])
        data[2::3] = bytes(g["B"])
        return Image.frombytes("RGB", ukuran, bytes(data))
    return Image.frombytes("L", ukuran, bytes(g["data"]))


# ------------------------- MENGGAMBAR DENGAN MATPLOTLIB ---------------------
def rapikan_ax(ax):
    ax.set_facecolor(BG_UTAMA)
    ax.set_xticks([])
    ax.set_yticks([])
    for sisi in ax.spines.values():
        sisi.set_visible(False)


def gambar_hasil(ax, hasil, tampilkan_ditolak=False):
    # potongan blister + elips hijau di setiap sisa obat (+ nomor & bentuk)
    rapikan_ax(ax)
    foto = hasil["foto"]
    ax.imshow(ke_gambar(foto))
    ukuran = max(foto["w"], foto["h"])
    if tampilkan_ditolak:
        for d in hasil["ditolak"]:
            ax.add_patch(Circle((d["cx"], d["cy"]), d["r"], fill=False, edgecolor=MERAH,
                                linewidth=1.3, linestyle="--"))
            ax.text(d["cx"], d["cy"] + d["r"] + ukuran * 0.012, d["alasan"], color="white", fontsize=7,
                    ha="center", va="top",
                    bbox={"facecolor": MERAH, "alpha": 0.8, "edgecolor": "none", "boxstyle": "round,pad=0.2"})
    for k, o in enumerate(hasil["obat"]):
        lebar = 2 * o["a"] + 5
        tinggi = 2 * o["b"] + 5
        ax.add_patch(Ellipse((o["cx"], o["cy"]), lebar, tinggi, angle=o["sudut"], fill=False,
                             edgecolor="black", linewidth=5, alpha=0.45))
        ax.add_patch(Ellipse((o["cx"], o["cy"]), lebar, tinggi, angle=o["sudut"], fill=False,
                             edgecolor=AKSEN, linewidth=2.5))
        ax.text(o["cx"], o["cy"], str(k + 1), color="white", fontsize=10, fontweight="bold",
                ha="center", va="center",
                bbox={"facecolor": AKSEN_GELAP, "edgecolor": "white", "linewidth": 1,
                      "boxstyle": "round,pad=0.25"})
    ax.set_xlim(-0.5, foto["w"] - 0.5)
    ax.set_ylim(foto["h"] - 0.5, -0.5)


def gambar_langkah(fig, langkah, hasil=None):
    g = langkah["gambar"]
    ax = fig.add_subplot(1, 1, 1)
    rapikan_ax(ax)
    if g["jenis"] == "hasil" and hasil is not None:
        gambar_hasil(ax, hasil, False)            # langkah terakhir = gambar hasil
    elif g["jenis"] == "rgb":
        ax.imshow(ke_gambar(g))
    elif g["jenis"] in ("gray", "mask"):
        ax.imshow(ke_gambar(g), cmap="gray", vmin=0, vmax=255)
    elif g["jenis"] == "heat":
        peta = ax.imshow(ke_gambar(g), cmap="magma", vmin=0, vmax=255)
        batang = fig.colorbar(peta, ax=ax, fraction=0.035, pad=0.02)
        batang.set_label(g.get("skala", ""), color=TEKS_REDUP, fontsize=8)
        batang.ax.tick_params(colors=TEKS_REDUP, labelsize=8)
        batang.outline.set_visible(False)
    elif g["jenis"] == "hist":
        gambar_histogram(ax, g)
    if not langkah.get("dipakai", True):
        ax.text(0.99, 0.98, "DILEWATI", transform=ax.transAxes, color="white", fontsize=11,
                fontweight="bold", ha="right", va="top",
                bbox={"facecolor": ABU, "edgecolor": "none", "boxstyle": "round,pad=0.4"})


def gambar_histogram(ax, g):
    nilai = g["hist"]
    ax.bar(range(len(nilai)), nilai, width=1.0, color="#60a5fa")
    ax.set_xlim(0, 255)
    ax.set_facecolor(BG_PANEL)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.tick_params(colors=TEKS_REDUP, labelsize=9)
    ax.set_xlabel(g.get("label", "nilai (0 - 255)"), color=TEKS_REDUP)
    ax.set_ylabel("jumlah piksel", color=TEKS_REDUP)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GARIS)
    if g.get("garis") is not None:
        ax.axvline(g["garis"], color=MERAH, linewidth=2)
        ax.text(g["garis"] + 3, ax.get_ylim()[1] * 0.95, g.get("nama_garis", "T") + " = " + str(g["garis"]),
                color=MERAH, fontsize=10, fontweight="bold", va="top")


# ------------------------------- JENDELA ------------------------------------
def tombol(induk, teks, perintah, warna_bg, warna_hover):
    b = tk.Button(induk, text=teks, command=perintah, bg=warna_bg, fg="white",
                  activebackground=warna_hover, activeforeground="white", relief="flat", bd=0,
                  padx=16, pady=7, cursor="hand2", font=(HURUF, 10, "bold"))
    b.bind("<Enter>", lambda e: b.config(bg=warna_hover) if b["state"] != "disabled" else None)
    b.bind("<Leave>", lambda e: b.config(bg=warna_bg) if b["state"] != "disabled" else None)
    return b


def buat_jendela(fungsi_analisis):
    root = tk.Tk()
    root.title("Deteksi Sisa Obat Blister - Digital Image Processing")
    root.geometry("1280x820")
    root.minsize(960, 640)
    root.configure(bg=BG_UTAMA)

    keadaan = {"path": "", "hasil": None, "indeks": -1, "item": []}
    tampil_ditolak = tk.BooleanVar(value=False)

    # ======================= BAR ATAS =======================
    atas = tk.Frame(root, bg=BG_PANEL, height=64)
    atas.pack(side="top", fill="x")
    atas.pack_propagate(False)
    judul = tk.Frame(atas, bg=BG_PANEL)
    judul.pack(side="left", padx=18)
    tk.Label(judul, text="●", fg=AKSEN, bg=BG_PANEL, font=(HURUF, 18)).pack(side="left")
    teks_judul = tk.Frame(judul, bg=BG_PANEL)
    teks_judul.pack(side="left", padx=(8, 0))
    tk.Label(teks_judul, text="Deteksi Sisa Obat", fg=TEKS, bg=BG_PANEL,
             font=(HURUF, 14, "bold")).pack(anchor="w")
    tk.Label(teks_judul, text="Digital Image Processing  ·  Python murni tanpa library & tanpa ML",
             fg=TEKS_REDUP, bg=BG_PANEL, font=(HURUF, 9)).pack(anchor="w")
    kanan_atas = tk.Frame(atas, bg=BG_PANEL)
    kanan_atas.pack(side="right", padx=18)
    label_file = tk.Label(kanan_atas, text="Belum ada gambar", fg=TEKS_REDUP, bg=BG_PANEL,
                          font=(HURUF, 10), anchor="e")
    label_file.pack(side="left", padx=(0, 14))
    tk.Frame(root, bg=GARIS, height=1).pack(side="top", fill="x")

    # ======================= STATUS (bawah) =======================
    bar_status = tk.Frame(root, bg=BG_PANEL, height=28)
    bar_status.pack(side="bottom", fill="x")
    bar_status.pack_propagate(False)
    label_status = tk.Label(bar_status, text="Pilih gambar blister, lalu tekan Deteksi.",
                            fg=TEKS_REDUP, bg=BG_PANEL, font=(HURUF, 9), anchor="w")
    label_status.pack(side="left", padx=14, fill="x", expand=True)
    tk.Label(bar_status, text="← / → : pindah langkah", fg=TEKS_REDUP, bg=BG_PANEL,
             font=(HURUF, 9)).pack(side="right", padx=14)

    tengah = tk.Frame(root, bg=BG_UTAMA)
    tengah.pack(side="top", fill="both", expand=True)

    # ======================= SIDEBAR (kiri, bisa digulir) =======================
    sidebar = tk.Frame(tengah, bg=BG_PANEL, width=300)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)
    tk.Label(sidebar, text="LANGKAH PENGOLAHAN", fg=TEKS_REDUP, bg=BG_PANEL,
             font=(HURUF, 9, "bold"), anchor="w").pack(fill="x", padx=16, pady=(14, 6))
    kanvas_daftar = tk.Canvas(sidebar, bg=BG_PANEL, highlightthickness=0, bd=0)
    kanvas_daftar.pack(side="left", fill="both", expand=True)
    daftar = tk.Frame(kanvas_daftar, bg=BG_PANEL)
    id_daftar = kanvas_daftar.create_window((0, 0), window=daftar, anchor="nw")
    daftar.bind("<Configure>", lambda e: kanvas_daftar.configure(scrollregion=kanvas_daftar.bbox("all")))
    kanvas_daftar.bind("<Configure>", lambda e: kanvas_daftar.itemconfigure(id_daftar, width=e.width))

    def gulir_mouse(event):
        kanvas_daftar.yview_scroll(int(-event.delta / 120), "units")

    kanvas_daftar.bind("<Enter>", lambda e: kanvas_daftar.bind_all("<MouseWheel>", gulir_mouse))
    kanvas_daftar.bind("<Leave>", lambda e: kanvas_daftar.unbind_all("<MouseWheel>"))
    tk.Frame(tengah, bg=GARIS, width=1).pack(side="left", fill="y")

    # ======================= PANEL UTAMA (kanan) =======================
    utama = tk.Frame(tengah, bg=BG_UTAMA)
    utama.pack(side="left", fill="both", expand=True, padx=18, pady=14)
    baris_judul = tk.Frame(utama, bg=BG_UTAMA)
    baris_judul.pack(side="top", fill="x")
    label_langkah = tk.Label(baris_judul, text="", fg=TEKS_REDUP, bg=BG_UTAMA,
                             font=(HURUF, 10, "bold"), anchor="w")
    label_langkah.pack(side="top", anchor="w")
    label_judul = tk.Label(baris_judul, text="Selamat datang", fg=TEKS, bg=BG_UTAMA,
                           font=(HURUF, 17, "bold"), anchor="w")
    label_judul.pack(side="left", anchor="w")
    navigasi = tk.Frame(baris_judul, bg=BG_UTAMA)
    navigasi.pack(side="right")

    kartu = tk.Frame(utama, bg=BG_KARTU, padx=18, pady=14)
    kartu.pack(side="bottom", fill="x", pady=(10, 0))
    fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_UTAMA)
    kanvas_fig = FigureCanvasTkAgg(fig, master=utama)
    widget_fig = kanvas_fig.get_tk_widget()
    widget_fig.configure(bg=BG_UTAMA, highlightthickness=0)
    widget_fig.pack(side="top", fill="both", expand=True, pady=(8, 0))

    baris_badge = tk.Frame(kartu, bg=BG_KARTU)
    baris_badge.pack(side="top", fill="x")
    badge_materi = tk.Label(baris_badge, text="", fg="white", bg=ABU, font=(HURUF, 9, "bold"), padx=10, pady=3)
    badge_status = tk.Label(baris_badge, text="", fg="white", bg=ABU, font=(HURUF, 9, "bold"), padx=10, pady=3)
    label_teknik = tk.Label(baris_badge, text="", fg=TEKS, bg=BG_KARTU, font=(HURUF, 12, "bold"),
                            anchor="w", justify="left")
    label_rumus = tk.Label(kartu, text="", fg="#93c5fd", bg=BG_UTAMA, font=(HURUF_MONO, 10),
                           anchor="w", justify="left", padx=10, pady=6)
    label_deskripsi = tk.Label(kartu, text="", fg=TEKS, bg=BG_KARTU, font=(HURUF, 10),
                               anchor="w", justify="left")
    label_jumlah = tk.Label(kartu, text="", fg=AKSEN, bg=BG_KARTU, font=(HURUF, 30, "bold"), anchor="w")
    cek_ditolak = tk.Checkbutton(kartu, text="Tampilkan region yang ditolak (lingkaran merah)",
                                 variable=tampil_ditolak, fg=TEKS, bg=BG_KARTU, selectcolor=BG_UTAMA,
                                 activebackground=BG_KARTU, activeforeground=TEKS, font=(HURUF, 10),
                                 bd=0, highlightthickness=0, cursor="hand2")

    def atur_lebar_teks(event):
        lebar = max(event.width - 40, 200)
        label_deskripsi.config(wraplength=lebar)
        label_rumus.config(wraplength=lebar)
        label_teknik.config(wraplength=max(lebar - 300, 150))

    kartu.bind("<Configure>", atur_lebar_teks)

    def kosongkan_kartu():
        for wdg in (badge_materi, badge_status, label_teknik, label_rumus, label_deskripsi,
                    label_jumlah, cek_ditolak):
            wdg.pack_forget()

    # ======================= ISI SIDEBAR =======================
    def buat_item(teks_nomor, teks_judul_item, teks_materi, warna, indeks, dipakai):
        item = tk.Frame(daftar, bg=BG_PANEL, cursor="hand2", padx=12, pady=7)
        item.pack(side="top", fill="x", padx=6, pady=1)
        nomor = tk.Label(item, text=teks_nomor, fg="white", bg=warna if dipakai else ABU,
                         font=(HURUF, 9, "bold"), width=3, pady=2)
        nomor.pack(side="left", anchor="n", pady=2)
        isi = tk.Frame(item, bg=BG_PANEL)
        isi.pack(side="left", fill="x", expand=True, padx=(10, 0))
        label1 = tk.Label(isi, text=teks_judul_item, fg=TEKS if dipakai else TEKS_REDUP, bg=BG_PANEL,
                          font=(HURUF, 10, "bold"), anchor="w", justify="left", wraplength=210)
        label1.pack(side="top", fill="x")
        label2 = tk.Label(isi, text=teks_materi + ("" if dipakai else "  ·  dilewati"), fg=TEKS_REDUP,
                          bg=BG_PANEL, font=(HURUF, 8), anchor="w", justify="left", wraplength=210)
        label2.pack(side="top", fill="x")
        semua = [item, isi, label1, label2]
        for wdg in semua + [nomor]:
            wdg.bind("<Button-1>", lambda e: tampilkan(indeks))
        keadaan["item"].append(semua)

    def isi_sidebar():
        for wdg in daftar.winfo_children():
            wdg.destroy()
        keadaan["item"] = []
        hasil = keadaan["hasil"]
        if hasil is None:
            tk.Label(daftar, text="Belum ada hasil.\nPilih gambar lalu tekan Deteksi.", fg=TEKS_REDUP,
                     bg=BG_PANEL, font=(HURUF, 10), justify="left").pack(anchor="w", padx=16, pady=10)
            return
        buat_item("★", "Hasil akhir", "Sisa obat: " + str(hasil["jumlah"]), AKSEN_GELAP, -1, True)
        for k, langkah in enumerate(hasil["langkah"]):
            buat_item(str(k + 1), langkah["judul"], langkah["materi"], warna_materi(langkah["materi"]),
                      k, langkah.get("dipakai", True))

    def sorot_item():
        posisi = keadaan["indeks"] + 1
        for k, semua in enumerate(keadaan["item"]):
            for wdg in semua:
                wdg.config(bg=BG_PILIH if k == posisi else BG_PANEL)

    # ======================= HALAMAN =======================
    def tampilkan_sambutan():
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        rapikan_ax(ax)
        ax.text(0.5, 0.56, "Pilih foto blister", color=TEKS, fontsize=22, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.45, "Tekan  \"Pilih Gambar\"  lalu  \"Deteksi\"  di kanan atas.\n"
                "Setiap langkah pengolahan citra akan tampil satu per satu.",
                color=TEKS_REDUP, fontsize=11, ha="center", va="center", transform=ax.transAxes)
        kanvas_fig.draw()
        label_langkah.config(text="")
        label_judul.config(text="Selamat datang")
        kosongkan_kartu()
        label_deskripsi.config(text="Aplikasi ini menghitung jumlah obat yang MASIH ADA di dalam blister "
                                    "dari sebuah foto, dengan pengolahan citra klasik (materi DIP).")
        label_deskripsi.pack(side="top", fill="x")

    def tampilkan_hasil():
        hasil = keadaan["hasil"]
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        gambar_hasil(ax, hasil, tampil_ditolak.get())
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        kanvas_fig.draw()
        label_langkah.config(text="HASIL AKHIR")
        label_judul.config(text=hasil["nama"])
        kosongkan_kartu()
        label_jumlah.config(text="SISA OBAT: " + str(hasil["jumlah"]))
        label_jumlah.pack(side="top", fill="x")
        per_bentuk = {}
        for o in hasil["obat"]:
            per_bentuk[o["bentuk"]] = per_bentuk.get(o["bentuk"], 0) + 1
        teks_bentuk = ", ".join(k + " " + str(v) for k, v in per_bentuk.items())
        dilewati = sum(1 for l in hasil["langkah"] if not l.get("dipakai", True))
        teks = ((("Bentuk: " + teks_bentuk + ".  ") if teks_bentuk else "") +
                "Elips hijau = obat yang masih ada (angka = nomor obat).  Region ditolak: " +
                str(len(hasil["ditolak"])) + ".  Langkah: " + str(len(hasil["langkah"])) +
                " (" + str(dilewati) + " dilewati karena tidak diperlukan).  Waktu proses: " +
                str(round(hasil["waktu"], 2)) + " detik.")
        label_deskripsi.config(text=teks)
        label_deskripsi.pack(side="top", fill="x", pady=(2, 6))
        cek_ditolak.pack(side="top", anchor="w")

    def tampilkan_langkah(i):
        hasil = keadaan["hasil"]
        langkah = hasil["langkah"][i]
        dipakai = langkah.get("dipakai", True)
        fig.clear()
        gambar_langkah(fig, langkah, hasil)
        if langkah["gambar"]["jenis"] == "hist":
            fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12)
        else:
            fig.subplots_adjust(left=0.01, right=0.95, top=0.99, bottom=0.01)
        kanvas_fig.draw()
        label_langkah.config(text="LANGKAH " + str(i + 1) + " DARI " + str(len(hasil["langkah"])))
        label_judul.config(text=langkah["judul"])
        kosongkan_kartu()
        badge_materi.config(text=langkah["materi"], bg=warna_materi(langkah["materi"]) if dipakai else ABU)
        badge_materi.pack(side="left")
        badge_status.config(text="DIPAKAI" if dipakai else "DILEWATI", bg=AKSEN_GELAP if dipakai else ABU)
        badge_status.pack(side="left", padx=(6, 0))
        label_teknik.config(text=langkah["teknik"])
        label_teknik.pack(side="left", padx=(12, 0))
        if langkah.get("rumus"):
            label_rumus.config(text=langkah["rumus"])
            label_rumus.pack(side="top", fill="x", pady=(10, 0))
        label_deskripsi.config(text=langkah["deskripsi"], fg=TEKS if dipakai else TEKS_REDUP)
        label_deskripsi.pack(side="top", fill="x", pady=(10, 0))

    def tampilkan(indeks):
        if keadaan["hasil"] is None:
            return
        keadaan["indeks"] = indeks
        label_deskripsi.config(fg=TEKS)
        if indeks < 0:
            tampilkan_hasil()
        else:
            tampilkan_langkah(indeks)
        sorot_item()
        tombol_kiri.config(state="disabled" if indeks <= -1 else "normal",
                           bg=BG_KARTU if indeks <= -1 else BG_PILIH)
        akhir = indeks >= len(keadaan["hasil"]["langkah"]) - 1
        tombol_kanan.config(state="disabled" if akhir else "normal", bg=BG_KARTU if akhir else BG_PILIH)

    def sebelumnya(event=None):
        if keadaan["hasil"] is not None and keadaan["indeks"] > -1:
            tampilkan(keadaan["indeks"] - 1)

    def berikutnya(event=None):
        if keadaan["hasil"] is not None and keadaan["indeks"] < len(keadaan["hasil"]["langkah"]) - 1:
            tampilkan(keadaan["indeks"] + 1)

    cek_ditolak.config(command=lambda: tampilkan_hasil() if keadaan["indeks"] == -1 else None)

    # ======================= AKSI TOMBOL =======================
    def pilih_gambar():
        path = filedialog.askopenfilename(
            title="Pilih foto blister",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG"), ("Semua file", "*.*")])
        if path:
            atur_path(path)

    def atur_path(path):
        keadaan["path"] = path
        label_file.config(text=os.path.basename(path), fg=TEKS)
        label_status.config(text="Gambar dipilih. Tekan Deteksi untuk memproses.", fg=TEKS_REDUP)

    def deteksi():
        if not keadaan["path"]:
            label_status.config(text="Pilih gambar dulu!", fg=MERAH)
            return
        label_status.config(text="Memproses... mohon tunggu", fg="#facc15")
        tombol_deteksi.config(state="disabled")
        root.config(cursor="watch")
        root.update()
        try:
            hasil = fungsi_analisis(keadaan["path"])
        except Exception as galat:
            root.config(cursor="")
            tombol_deteksi.config(state="normal")
            label_status.config(text="Gagal memproses: " + str(galat), fg=MERAH)
            messagebox.showerror("Gagal", "Terjadi kesalahan saat memproses:\n\n" + str(galat))
            return
        root.config(cursor="")
        tombol_deteksi.config(state="normal")
        keadaan["hasil"] = hasil
        isi_sidebar()
        kanvas_daftar.yview_moveto(0)
        tampilkan(-1)
        label_status.config(text="Selesai: sisa obat = " + str(hasil["jumlah"]) + "   (" +
                            str(round(hasil["waktu"], 2)) + " detik, " + str(len(hasil["langkah"])) +
                            " langkah)", fg=AKSEN)

    def simpan_hasil():
        # simpan gambar hasil + log penjelasan semua langkah ke folder hasil/
        hasil = keadaan["hasil"]
        if hasil is None:
            label_status.config(text="Belum ada hasil untuk disimpan.", fg=MERAH)
            return
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil")
        os.makedirs(folder, exist_ok=True)
        dasar = os.path.splitext(hasil["nama"])[0].replace(" ", "_")
        fig_simpan = Figure(figsize=(8, 8), dpi=110, facecolor=BG_UTAMA)
        gambar_hasil(fig_simpan.add_subplot(1, 1, 1), hasil, False)
        fig_simpan.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        fig_simpan.savefig(os.path.join(folder, dasar + "_hasil.png"), facecolor=BG_UTAMA)
        with open(os.path.join(folder, dasar + "_langkah.txt"), "w", encoding="utf-8") as berkas:
            berkas.write(hasil["nama"] + "\nSISA OBAT = " + str(hasil["jumlah"]) + "\n\n")
            for k, langkah in enumerate(hasil["langkah"]):
                berkas.write(str(k + 1) + ". " + langkah["judul"] + "  [" + langkah["materi"] + "]" +
                             ("" if langkah.get("dipakai", True) else "  (dilewati)") + "\n")
                berkas.write("   Teknik : " + langkah["teknik"] + "\n")
                berkas.write("   Rumus  : " + langkah["rumus"] + "\n")
                berkas.write("   " + langkah["deskripsi"] + "\n\n")
        label_status.config(text="Disimpan ke folder hasil/: " + dasar + "_hasil.png dan " + dasar +
                            "_langkah.txt", fg=AKSEN)

    tombol_pilih = tombol(kanan_atas, "Pilih Gambar", pilih_gambar, BG_PILIH, "#3a4356")
    tombol_pilih.pack(side="left", padx=(0, 8))
    tombol_deteksi = tombol(kanan_atas, "Deteksi", deteksi, AKSEN_GELAP, AKSEN)
    tombol_deteksi.pack(side="left", padx=(0, 8))
    tombol_simpan = tombol(kanan_atas, "Simpan Hasil", simpan_hasil, BG_PILIH, "#3a4356")
    tombol_simpan.pack(side="left")
    tombol_kiri = tombol(navigasi, "◀  Sebelumnya", sebelumnya, BG_PILIH, "#3a4356")
    tombol_kiri.pack(side="left", padx=(0, 6))
    tombol_kanan = tombol(navigasi, "Berikutnya  ▶", berikutnya, BG_PILIH, "#3a4356")
    tombol_kanan.pack(side="left")
    tombol_kiri.config(state="disabled", bg=BG_KARTU, disabledforeground=TEKS_REDUP)
    tombol_kanan.config(state="disabled", bg=BG_KARTU, disabledforeground=TEKS_REDUP)
    root.bind("<Left>", sebelumnya)
    root.bind("<Right>", berikutnya)

    isi_sidebar()
    tampilkan_sambutan()
    # disimpan di root supaya bisa dipanggil dari luar (uji otomatis)
    root.atur_path = atur_path
    root.deteksi = deteksi
    root.tampilkan = tampilkan
    root.simpan_hasil = simpan_hasil
    root.fig = fig
    return root


def jalankan_jendela(fungsi_analisis, path_awal=None):
    root = buat_jendela(fungsi_analisis)
    if path_awal is not None:
        root.atur_path(path_awal)
        root.after(200, root.deteksi)
    root.mainloop()
