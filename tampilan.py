# ============================================================================
# tampilan.py  -  Jendela (GUI) untuk deteksi sisa obat pada blister
#
# Cara pakai (dari main.py):
#     from tampilan import jalankan_jendela
#     jalankan_jendela(analisis)        # analisis(path) -> dict hasil
#
# Library: tkinter (jendela), matplotlib (gambar), numpy (array)
# ============================================================================
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ------------------------------- WARNA TEMA ---------------------------------
BG_UTAMA = "#14171f"      # latar jendela
BG_PANEL = "#1c2029"      # latar sidebar / kartu
BG_KARTU = "#232834"      # latar kartu deskripsi
BG_PILIH = "#2c3342"      # item sidebar yang dipilih
GARIS = "#323949"         # garis pemisah
TEKS = "#e6e9ef"          # teks utama
TEKS_REDUP = "#8b93a5"    # teks kedua
AKSEN = "#22c55e"         # hijau = warna aksen (sama dengan lingkaran obat)
AKSEN_GELAP = "#16a34a"
MERAH = "#ef4444"
ABU = "#4b5263"           # badge "dilewati"

HURUF = "Segoe UI"
HURUF_MONO = "Consolas"

# warna badge per materi (berdasarkan nomor materi di depan nama)
WARNA_MATERI = {
    "01": "#64748b",
    "02": "#0ea5e9",
    "03": "#8b5cf6",
    "04": "#f59e0b",
    "05": "#ec4899",
    "06": "#14b8a6",
    "07": "#a3a3a3",
    "08": "#3b82f6",
    "09": "#f97316",
    "10": "#84cc16",
}
WARNA_CADANGAN = ["#06b6d4", "#d946ef", "#eab308", "#10b981", "#f43f5e"]


def warna_materi(materi):
    # ambil 2 huruf pertama, mis. "08 Image Segmentation" -> "08"
    kode = materi[:2]
    if kode in WARNA_MATERI:
        return WARNA_MATERI[kode]
    # materi lain: pilih warna cadangan dari jumlah kode huruf
    total = 0
    for huruf in materi:
        total = total + ord(huruf)
    return WARNA_CADANGAN[total % len(WARNA_CADANGAN)]


# ------------------------- MENGGAMBAR DENGAN MATPLOTLIB ---------------------
def rapikan_ax(ax):
    # sumbu tanpa garis/angka, latar gelap
    ax.set_facecolor(BG_UTAMA)
    ax.set_xticks([])
    ax.set_yticks([])
    for sisi in ax.spines.values():
        sisi.set_visible(False)


def ke_rgb(gambar):
    # gambar 0..255 -> 0..1 supaya bisa ditampilkan imshow
    return np.clip(np.asarray(gambar, dtype=float) / 255.0, 0, 1)


def label_nomor(nomor, butir):
    # obat ke-3 berisi 2 butir -> "3-4"
    if butir > 1:
        return str(nomor) + "-" + str(nomor + butir - 1)
    return str(nomor)


def gambar_hasil(ax, hasil, tampilkan_ditolak=False):
    # Foto blister + lingkaran hijau di setiap sisa obat.
    # Bila tampilkan_ditolak=True, region yang ditolak diberi lingkaran merah putus-putus.
    rapikan_ax(ax)
    foto = hasil["foto"]
    ax.imshow(ke_rgb(foto))
    tinggi = foto.shape[0]
    lebar = foto.shape[1]
    ukuran = max(tinggi, lebar)

    # region yang ditolak (merah, tipis, putus-putus)
    if tampilkan_ditolak:
        for d in hasil["ditolak"]:
            lingkaran = Circle((d["cx"], d["cy"]), d["r"], fill=False,
                               edgecolor=MERAH, linewidth=1.5, linestyle="--")
            ax.add_patch(lingkaran)
            ax.text(d["cx"], d["cy"] + d["r"] + ukuran * 0.012, d["alasan"],
                    color="white", fontsize=7, ha="center", va="top",
                    bbox={"facecolor": MERAH, "alpha": 0.8, "edgecolor": "none",
                          "boxstyle": "round,pad=0.2"})

    # sisa obat (hijau, tebal) + nomor
    nomor = 1
    for o in hasil["obat"]:
        butir = int(o.get("butir", 1))
        # lingkaran hitam tipis di belakang supaya hijau terlihat di latar terang
        bayangan = Circle((o["cx"], o["cy"]), o["r"], fill=False,
                          edgecolor="black", linewidth=5, alpha=0.45)
        ax.add_patch(bayangan)
        lingkaran = Circle((o["cx"], o["cy"]), o["r"], fill=False,
                           edgecolor=AKSEN, linewidth=3)
        ax.add_patch(lingkaran)
        teks_label = label_nomor(nomor, butir)
        if "bentuk" in o:
            teks_label = teks_label + " " + o["bentuk"]
        ax.text(o["cx"], o["cy"] - o["r"], teks_label,
                color="white", fontsize=10, fontweight="bold",
                ha="center", va="center",
                bbox={"facecolor": AKSEN_GELAP, "edgecolor": "white",
                      "linewidth": 1, "boxstyle": "round,pad=0.25"})
        nomor = nomor + butir

    ax.set_xlim(-0.5, lebar - 0.5)
    ax.set_ylim(tinggi - 0.5, -0.5)


def gambar_langkah(fig, langkah):
    # gambar satu langkah sesuai "jenis"-nya
    jenis = langkah["jenis"]
    ax = fig.add_subplot(1, 1, 1)
    rapikan_ax(ax)
    gambar = langkah.get("gambar")

    if jenis == "rgb":
        ax.imshow(ke_rgb(gambar))
    elif jenis == "gray":
        ax.imshow(gambar, cmap="gray")
    elif jenis == "mask":
        ax.imshow(gambar, cmap="gray", vmin=0, vmax=1)
    elif jenis == "heat":
        peta = ax.imshow(gambar, cmap="magma")
        batang = fig.colorbar(peta, ax=ax, fraction=0.035, pad=0.02)
        batang.ax.tick_params(colors=TEKS_REDUP, labelsize=8)
        batang.outline.set_visible(False)
    elif jenis == "spektrum":
        # log(1 + |F|) supaya puncak kecil ikut terlihat
        ax.imshow(np.log(1 + np.abs(np.asarray(gambar, dtype=float))), cmap="gray")
    elif jenis == "hist":
        gambar_histogram(ax, langkah)
    else:
        # jenis "none": tidak ada gambar, cukup teks
        ax.text(0.5, 0.5, "Langkah ini tidak menghasilkan gambar.\nLihat penjelasan di bawah.",
                color=TEKS_REDUP, fontsize=13, ha="center", va="center",
                transform=ax.transAxes)

    # tanda besar bila langkah dilewati
    if not langkah.get("dipakai", True):
        ax.text(0.99, 0.98, "DILEWATI", transform=ax.transAxes,
                color="white", fontsize=11, fontweight="bold", ha="right", va="top",
                bbox={"facecolor": ABU, "edgecolor": "none", "boxstyle": "round,pad=0.4"})


def gambar_histogram(ax, langkah):
    nilai = langkah.get("histogram")
    if nilai is None:
        nilai = [0] * 256
    ax.bar(range(len(nilai)), nilai, width=1.0, color="#60a5fa")
    ax.set_xlim(0, 255)
    ax.set_facecolor(BG_PANEL)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.tick_params(colors=TEKS_REDUP, labelsize=9)
    ax.set_xlabel("intensitas (0 - 255)", color=TEKS_REDUP)
    ax.set_ylabel("jumlah piksel", color=TEKS_REDUP)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(GARIS)
    garis = langkah.get("garis")
    if garis is not None:
        ax.axvline(garis, color=MERAH, linewidth=2)
        ax.text(garis + 3, ax.get_ylim()[1] * 0.95, "T = " + str(garis),
                color=MERAH, fontsize=10, fontweight="bold", va="top")


# ------------------------------- JENDELA ------------------------------------
def tombol(induk, teks, perintah, warna_bg, warna_hover, lebar=None):
    # tombol datar dengan efek hover
    b = tk.Button(induk, text=teks, command=perintah, bg=warna_bg, fg="white",
                  activebackground=warna_hover, activeforeground="white",
                  relief="flat", bd=0, padx=16, pady=7, cursor="hand2",
                  font=(HURUF, 10, "bold"))
    if lebar is not None:
        b.config(width=lebar)

    def masuk(event):
        if b["state"] != "disabled":
            b.config(bg=warna_hover)

    def keluar(event):
        b.config(bg=warna_bg)

    b.bind("<Enter>", masuk)
    b.bind("<Leave>", keluar)
    return b


def buat_jendela(fungsi_analisis):
    # Membuat semua bagian jendela, mengembalikan root (belum mainloop)
    root = tk.Tk()
    root.title("Deteksi Sisa Obat Blister - Digital Image Processing")
    root.geometry("1280x800")
    root.minsize(960, 620)
    root.configure(bg=BG_UTAMA)

    # data yang berubah selama program berjalan
    keadaan = {
        "path": "",          # path gambar terpilih
        "hasil": None,       # dict hasil analisis
        "indeks": -1,        # -1 = halaman Hasil, 0.. = langkah ke-i
        "item": [],          # widget item di sidebar
    }
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
    tk.Label(teks_judul, text="Digital Image Processing  ·  blister pack", fg=TEKS_REDUP,
             bg=BG_PANEL, font=(HURUF, 9)).pack(anchor="w")

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

    # ======================= SIDEBAR (kiri) =======================
    sidebar = tk.Frame(tengah, bg=BG_PANEL, width=290)
    sidebar.pack(side="left", fill="y")
    sidebar.pack_propagate(False)
    tk.Label(sidebar, text="LANGKAH PENGOLAHAN", fg=TEKS_REDUP, bg=BG_PANEL,
             font=(HURUF, 9, "bold"), anchor="w").pack(fill="x", padx=16, pady=(14, 6))

    # daftar langkah bisa panjang -> pakai Canvas yang bisa digulir dengan roda mouse
    kanvas_daftar = tk.Canvas(sidebar, bg=BG_PANEL, highlightthickness=0, bd=0)
    kanvas_daftar.pack(side="left", fill="both", expand=True)
    daftar = tk.Frame(kanvas_daftar, bg=BG_PANEL)
    id_daftar = kanvas_daftar.create_window((0, 0), window=daftar, anchor="nw")

    def daftar_berubah(event):
        kanvas_daftar.configure(scrollregion=kanvas_daftar.bbox("all"))

    def kanvas_berubah(event):
        kanvas_daftar.itemconfigure(id_daftar, width=event.width)

    daftar.bind("<Configure>", daftar_berubah)
    kanvas_daftar.bind("<Configure>", kanvas_berubah)

    def gulir_mouse(event):
        kanvas_daftar.yview_scroll(int(-event.delta / 120), "units")

    def mouse_masuk(event):
        kanvas_daftar.bind_all("<MouseWheel>", gulir_mouse)

    def mouse_keluar(event):
        kanvas_daftar.unbind_all("<MouseWheel>")

    # roda mouse hanya menggulir sidebar saat kursor di atasnya
    kanvas_daftar.bind("<Enter>", mouse_masuk)
    kanvas_daftar.bind("<Leave>", mouse_keluar)

    tk.Frame(tengah, bg=GARIS, width=1).pack(side="left", fill="y")

    # ======================= PANEL UTAMA (kanan) =======================
    utama = tk.Frame(tengah, bg=BG_UTAMA)
    utama.pack(side="left", fill="both", expand=True, padx=18, pady=14)

    # baris judul panel + tombol navigasi
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

    # kartu deskripsi (bawah gambar)
    kartu = tk.Frame(utama, bg=BG_KARTU, padx=18, pady=14)
    kartu.pack(side="bottom", fill="x", pady=(10, 0))

    # gambar matplotlib (mengisi sisa ruang)
    fig = Figure(figsize=(8, 5), dpi=100, facecolor=BG_UTAMA)
    kanvas_fig = FigureCanvasTkAgg(fig, master=utama)
    widget_fig = kanvas_fig.get_tk_widget()
    widget_fig.configure(bg=BG_UTAMA, highlightthickness=0)
    widget_fig.pack(side="top", fill="both", expand=True, pady=(8, 0))

    # --- isi kartu ---
    baris_badge = tk.Frame(kartu, bg=BG_KARTU)
    baris_badge.pack(side="top", fill="x")
    badge_materi = tk.Label(baris_badge, text="", fg="white", bg=ABU,
                            font=(HURUF, 9, "bold"), padx=10, pady=3)
    badge_status = tk.Label(baris_badge, text="", fg="white", bg=ABU,
                            font=(HURUF, 9, "bold"), padx=10, pady=3)
    label_teknik = tk.Label(baris_badge, text="", fg=TEKS, bg=BG_KARTU,
                            font=(HURUF, 12, "bold"), anchor="w", justify="left")

    label_rumus = tk.Label(kartu, text="", fg="#93c5fd", bg=BG_UTAMA,
                           font=(HURUF_MONO, 11), anchor="w", justify="left",
                           padx=10, pady=6)
    label_deskripsi = tk.Label(kartu, text="", fg=TEKS, bg=BG_KARTU,
                               font=(HURUF, 10), anchor="w", justify="left")

    # kartu khusus halaman Hasil
    label_jumlah = tk.Label(kartu, text="", fg=AKSEN, bg=BG_KARTU,
                            font=(HURUF, 30, "bold"), anchor="w")
    cek_ditolak = tk.Checkbutton(kartu, text="Tampilkan region yang ditolak (lingkaran merah)",
                                 variable=tampil_ditolak, fg=TEKS, bg=BG_KARTU,
                                 selectcolor=BG_UTAMA, activebackground=BG_KARTU,
                                 activeforeground=TEKS, font=(HURUF, 10),
                                 bd=0, highlightthickness=0, cursor="hand2")

    def atur_lebar_teks(event):
        # teks deskripsi dibungkus mengikuti lebar kartu
        lebar = event.width - 40
        if lebar < 200:
            lebar = 200
        label_deskripsi.config(wraplength=lebar)
        label_rumus.config(wraplength=lebar)
        # teknik di sebelah 2 badge -> sisa lebar dikurangi +-280 px
        label_teknik.config(wraplength=max(lebar - 280, 150))

    kartu.bind("<Configure>", atur_lebar_teks)

    def kosongkan_kartu():
        for w in [badge_materi, badge_status, label_teknik, label_rumus,
                  label_deskripsi, label_jumlah, cek_ditolak]:
            w.pack_forget()

    # ======================= ISI SIDEBAR =======================
    def buat_item(teks_nomor, teks_judul_item, teks_materi, warna, indeks, dipakai):
        # satu baris di sidebar: [nomor] judul / badge materi
        item = tk.Frame(daftar, bg=BG_PANEL, cursor="hand2", padx=12, pady=8)
        item.pack(side="top", fill="x", padx=6, pady=1)

        warna_nomor = warna
        if not dipakai:
            warna_nomor = ABU
        nomor = tk.Label(item, text=teks_nomor, fg="white", bg=warna_nomor,
                         font=(HURUF, 9, "bold"), width=3, pady=2)
        nomor.pack(side="left", anchor="n", pady=2)

        isi = tk.Frame(item, bg=BG_PANEL)
        isi.pack(side="left", fill="x", expand=True, padx=(10, 0))
        warna_teks = TEKS
        if not dipakai:
            warna_teks = TEKS_REDUP
        label1 = tk.Label(isi, text=teks_judul_item, fg=warna_teks, bg=BG_PANEL,
                          font=(HURUF, 10, "bold"), anchor="w", justify="left",
                          wraplength=200)
        label1.pack(side="top", fill="x")
        teks_bawah = teks_materi
        if not dipakai:
            teks_bawah = teks_materi + "  ·  dilewati"
        label2 = tk.Label(isi, text=teks_bawah, fg=TEKS_REDUP, bg=BG_PANEL,
                          font=(HURUF, 8), anchor="w")
        label2.pack(side="top", fill="x")

        semua = [item, isi, label1, label2]

        def klik(event):
            tampilkan(indeks)

        for w in semua + [nomor]:
            w.bind("<Button-1>", klik)
        keadaan["item"].append(semua)

    def isi_sidebar():
        for w in daftar.winfo_children():
            w.destroy()
        keadaan["item"] = []
        hasil = keadaan["hasil"]
        if hasil is None:
            tk.Label(daftar, text="Belum ada hasil.\nPilih gambar lalu tekan Deteksi.",
                     fg=TEKS_REDUP, bg=BG_PANEL, font=(HURUF, 10),
                     justify="left").pack(anchor="w", padx=16, pady=10)
            return
        buat_item("★", "Hasil akhir", "Sisa obat: " + str(hasil["jumlah"]),
                  AKSEN_GELAP, -1, True)
        nomor = 1
        for langkah in hasil["langkah"]:
            buat_item(str(nomor), langkah["judul"], langkah["materi"],
                      warna_materi(langkah["materi"]), nomor - 1,
                      langkah.get("dipakai", True))
            nomor = nomor + 1

    def sorot_item():
        # beri warna latar pada item yang sedang dipilih
        posisi = keadaan["indeks"] + 1          # item 0 = Hasil
        i = 0
        for semua in keadaan["item"]:
            warna = BG_PANEL
            if i == posisi:
                warna = BG_PILIH
            for w in semua:
                w.config(bg=warna)
            i = i + 1

    # ======================= MENAMPILKAN HALAMAN =======================
    def tampilkan_sambutan():
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        rapikan_ax(ax)
        ax.text(0.5, 0.56, "Pilih foto blister", color=TEKS, fontsize=22,
                fontweight="bold", ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.45, "Tekan  \"Pilih Gambar\"  lalu  \"Deteksi\"  di kanan atas.\n"
                "Setiap langkah pengolahan citra akan tampil satu per satu.",
                color=TEKS_REDUP, fontsize=11, ha="center", va="center",
                transform=ax.transAxes)
        kanvas_fig.draw()
        label_langkah.config(text="")
        label_judul.config(text="Selamat datang")
        kosongkan_kartu()
        label_deskripsi.config(text="Aplikasi ini menghitung jumlah obat yang MASIH ADA "
                               "di dalam blister dari sebuah foto.")
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

        # ringkasan: berapa langkah dipakai / dilewati
        dipakai = len(hasil["langkah"])
        dilewati = hasil.get("jumlah_dilewati", 0)
        # jumlah obat per bentuk, mis. "bulat 4, kapsul 2"
        per_bentuk = {}
        for o in hasil["obat"]:
            nama_bentuk = o.get("bentuk", "-")
            if nama_bentuk in per_bentuk:
                per_bentuk[nama_bentuk] += int(o.get("butir", 1))
            else:
                per_bentuk[nama_bentuk] = int(o.get("butir", 1))
        daftar_bentuk = []
        for nama_bentuk in per_bentuk:
            daftar_bentuk.append(nama_bentuk + " " + str(per_bentuk[nama_bentuk]))
        teks_bentuk = ""
        if len(daftar_bentuk) > 0:
            teks_bentuk = "Bentuk: " + ", ".join(daftar_bentuk) + ".  "
        teks = (teks_bentuk + "Lingkaran hijau = obat yang masih ada (angka = nomor obat; \"3-4\" berarti "
                "satu region berisi 2 butir).  Region ditolak: " + str(len(hasil["ditolak"])) +
                ".  Langkah dipakai: " + str(dipakai) + ", dilewati: " + str(dilewati) +
                ".  Waktu proses: " + str(round(hasil["waktu"], 2)) + " detik.")
        label_deskripsi.config(text=teks)
        label_deskripsi.pack(side="top", fill="x", pady=(2, 6))
        cek_ditolak.pack(side="top", anchor="w")

    def tampilkan_langkah(i):
        hasil = keadaan["hasil"]
        langkah = hasil["langkah"][i]
        dipakai = langkah.get("dipakai", True)

        fig.clear()
        gambar_langkah(fig, langkah)
        if langkah["jenis"] == "hist":
            fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12)
        else:
            fig.subplots_adjust(left=0.01, right=0.95, top=0.99, bottom=0.01)
        kanvas_fig.draw()

        label_langkah.config(text="LANGKAH " + str(i + 1) + " DARI " + str(len(hasil["langkah"])))
        label_judul.config(text=langkah["judul"])
        kosongkan_kartu()

        warna = warna_materi(langkah["materi"])
        if not dipakai:
            warna = ABU
        badge_materi.config(text=langkah["materi"], bg=warna)
        badge_materi.pack(side="left")
        if dipakai:
            badge_status.config(text="DIPAKAI", bg=AKSEN_GELAP)
        else:
            badge_status.config(text="DILEWATI", bg=ABU)
        badge_status.pack(side="left", padx=(6, 0))
        label_teknik.config(text=langkah["teknik"])
        label_teknik.pack(side="left", padx=(12, 0))

        if langkah.get("rumus", "") != "":
            label_rumus.config(text=langkah["rumus"])
            label_rumus.pack(side="top", fill="x", pady=(10, 0))
        warna_teks = TEKS
        if not dipakai:
            warna_teks = TEKS_REDUP
        label_deskripsi.config(text=langkah["deskripsi"], fg=warna_teks)
        label_deskripsi.pack(side="top", fill="x", pady=(10, 0))

    def tampilkan(indeks):
        # indeks -1 = Hasil, 0.. = langkah ke-indeks
        if keadaan["hasil"] is None:
            return
        keadaan["indeks"] = indeks
        label_deskripsi.config(fg=TEKS)
        if indeks < 0:
            tampilkan_hasil()
        else:
            tampilkan_langkah(indeks)
        sorot_item()
        # tombol navigasi aktif / tidak
        if indeks <= -1:
            tombol_kiri.config(state="disabled", bg=BG_KARTU)
        else:
            tombol_kiri.config(state="normal", bg=BG_PILIH)
        if indeks >= len(keadaan["hasil"]["langkah"]) - 1:
            tombol_kanan.config(state="disabled", bg=BG_KARTU)
        else:
            tombol_kanan.config(state="normal", bg=BG_PILIH)

    def sebelumnya(event=None):
        if keadaan["hasil"] is not None and keadaan["indeks"] > -1:
            tampilkan(keadaan["indeks"] - 1)

    def berikutnya(event=None):
        if keadaan["hasil"] is None:
            return
        if keadaan["indeks"] < len(keadaan["hasil"]["langkah"]) - 1:
            tampilkan(keadaan["indeks"] + 1)

    def ganti_ditolak():
        if keadaan["hasil"] is not None and keadaan["indeks"] == -1:
            tampilkan_hasil()

    cek_ditolak.config(command=ganti_ditolak)

    # ======================= AKSI TOMBOL =======================
    def pilih_gambar():
        path = filedialog.askopenfilename(
            title="Pilih foto blister",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG"),
                       ("Semua file", "*.*")])
        if path != "":
            atur_path(path)

    def atur_path(path):
        keadaan["path"] = path
        nama = path.replace("\\", "/").split("/")[-1]
        label_file.config(text=nama, fg=TEKS)
        label_status.config(text="Gambar dipilih. Tekan Deteksi untuk memproses.", fg=TEKS_REDUP)

    def deteksi():
        if keadaan["path"] == "":
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

        # hanya langkah yang DIPAKAI yang ditampilkan;
        # langkah yang dilewati cukup dihitung jumlahnya
        dipakai = []
        dilewati = 0
        for langkah in hasil["langkah"]:
            if langkah.get("dipakai", True):
                dipakai.append(langkah)
            else:
                dilewati = dilewati + 1
        hasil["langkah"] = dipakai
        hasil["jumlah_dilewati"] = dilewati

        keadaan["hasil"] = hasil
        isi_sidebar()
        kanvas_daftar.yview_moveto(0)
        tampilkan(-1)
        label_status.config(text="Selesai: sisa obat = " + str(hasil["jumlah"]) + "   (" +
                            str(round(hasil["waktu"], 2)) + " detik, " +
                            str(len(hasil["langkah"])) + " langkah)", fg=AKSEN)

    # tombol di bar atas
    tombol_pilih = tombol(kanan_atas, "Pilih Gambar", pilih_gambar, BG_PILIH, "#3a4356")
    tombol_pilih.pack(side="left", padx=(0, 8))
    tombol_deteksi = tombol(kanan_atas, "Deteksi", deteksi, AKSEN_GELAP, AKSEN)
    tombol_deteksi.pack(side="left")

    # tombol navigasi
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

    # disimpan di root supaya bisa dipanggil dari luar (mis. untuk uji coba)
    root.atur_path = atur_path
    root.deteksi = deteksi
    root.tampilkan = tampilkan
    return root


def jalankan_jendela(fungsi_analisis, path_awal=None):
    # fungsi_analisis(path) -> dict hasil (lihat kontrak data)
    # path_awal (opsional): gambar yang langsung diproses saat jendela dibuka
    root = buat_jendela(fungsi_analisis)
    if path_awal is not None:
        root.atur_path(path_awal)
        root.after(200, root.deteksi)
    root.mainloop()
