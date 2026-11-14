# =============================================================================
# PEMBUAT GAMBAR UJI (sintetis) untuk deteksi_sisa_obat.py
# Membuat blister buatan + variasinya (gelap, terang, kontras rendah,
# salt & pepper, gaussian noise, noise sinus) supaya setiap cabang
# perbaikan otomatis bisa dicoba. Jumlah sisa obat yang benar ditulis
# ke gambar/jawaban.txt
#
# CATATAN: ini hanya alat uji. Foto blister ASLI tetap harus dicoba.
# Cara pakai: python buat_contoh.py
# =============================================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gambar")
acak = np.random.default_rng(7)


def haluskan(a, kali):
    for k in range(kali):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5
    return a


def buat_blister(lebar, tinggi, baris, kolom, isi, jenis_obat, warna_obat, warna_meja, warna_foil):
    img = np.zeros((tinggi, lebar, 3))
    yy = np.zeros((tinggi, lebar))
    xx = np.zeros((tinggi, lebar))
    for y in range(tinggi):
        yy[y, :] = y
    for x in range(lebar):
        xx[:, x] = x

    # meja: warna rata + gradasi cahaya + serat halus
    serat = haluskan(acak.normal(0, 6, (tinggi, lebar)), 2)
    for c in range(3):
        img[:, :, c] = warna_meja[c] * (0.9 + 0.2 * xx / lebar) + serat

    # foil: perak dengan kerutan acak (tekstur kasar)
    x0 = int(lebar * 0.12)
    x1 = int(lebar * 0.88)
    y0 = int(tinggi * 0.15)
    y1 = int(tinggi * 0.85)
    kerut = haluskan(acak.normal(0, 30, (tinggi, lebar)), 1)
    garis = 8 * np.sin(xx * 0.9 + yy * 0.4)
    foil = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    for c in range(3):
        nilai = warna_foil[c] + kerut + garis
        img[:, :, c][foil] = nilai[foil]

    jarak_x = (x1 - x0) / kolom
    jarak_y = (y1 - y0) / baris
    if jenis_obat == "bulat":
        rx = jarak_x * 0.30
        ry = rx
    else:                                   # kapsul
        rx = jarak_x * 0.38
        ry = jarak_y * 0.20

    nomor = 0
    for b in range(baris):
        for k in range(kolom):
            cx = x0 + (k + 0.5) * jarak_x
            cy = y0 + (b + 0.5) * jarak_y
            d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            kantong = d < 1.35
            pinggir = (d >= 1.0) & (d < 1.35)
            status = isi[nomor]
            nomor += 1
            if status == "isi":
                obat = d < 1.0
                for c in range(3):
                    terang = warna_obat[c] * (1 - 0.12 * d) + acak.normal(0, 2, (tinggi, lebar))
                    img[:, :, c][obat] = terang[obat]
                    img[:, :, c][pinggir] = 235          # kilau plastik
            elif status == "kosong":                    # plastik kusut, berkilau
                kusut = haluskan(acak.normal(0, 55, (tinggi, lebar)), 1)
                for c in range(3):
                    nilai = 150 + kusut
                    img[:, :, c][kantong] = nilai[kantong]
            else:                                       # foil sobek -> terlihat meja
                sobek = d < 1.0 + 0.25 * np.sin(xx * 1.3) * np.cos(yy * 1.7)
                for c in range(3):
                    nilai = warna_meja[c] * (0.9 + 0.2 * xx / lebar) + serat
                    img[:, :, c][sobek] = nilai[sobek]
                    img[:, :, c][pinggir & ~sobek] = 230
    return np.clip(img, 0, 255)


def simpan(nama, img):
    plt.imsave(os.path.join(FOLDER, nama), img.astype(np.uint8))


os.makedirs(FOLDER, exist_ok=True)
jawaban = []

isi1 = ["isi", "kosong", "isi", "isi", "kosong",
        "kosong", "isi", "sobek", "kosong", "isi"]
c1 = buat_blister(520, 380, 2, 5, isi1, "bulat", (240, 238, 232), (150, 110, 75), (175, 178, 182))
simpan("contoh1.png", c1)
jawaban.append("contoh1.png 5")

isi2 = ["isi", "isi", "kosong", "sobek",
        "kosong", "isi", "isi", "kosong",
        "isi", "kosong", "kosong", "isi"]
c2 = buat_blister(560, 420, 3, 4, isi2, "kapsul", (230, 120, 40), (60, 70, 90), (190, 190, 195))
simpan("contoh2.png", c2)
jawaban.append("contoh2.png 6")

isi3 = ["kosong", "kosong", "isi", "kosong", "kosong",
        "kosong", "sobek", "kosong", "kosong", "isi"]
c3 = buat_blister(480, 360, 2, 5, isi3, "bulat", (200, 60, 70), (215, 210, 200), (160, 165, 170))
simpan("contoh3.png", c3)
jawaban.append("contoh3.png 2")

# ---------------- variasi dari contoh1 (jawaban tetap 5) ----------------
tinggi = c1.shape[0]
lebar = c1.shape[1]
simpan("contoh1_gelap.png", c1 * 0.30)
simpan("contoh1_terang.png", 255 * (c1 / 255) ** 0.35)
simpan("contoh1_kontrasrendah.png", 90 + c1 * 0.30)

sp = c1.copy()
peluang = acak.random((tinggi, lebar))
sp[peluang < 0.03] = 0
sp[peluang > 0.97] = 255
simpan("contoh1_saltpepper.png", sp)

simpan("contoh1_gaussian.png", np.clip(c1 + acak.normal(0, 20, c1.shape), 0, 255))

sinus = np.zeros((tinggi, lebar))
for y in range(tinggi):
    for x in range(lebar):
        sinus[y, x] = 35 * math.sin(2 * math.pi * (x / 7 + y / 11))
sin_img = c1.copy()
for c in range(3):
    sin_img[:, :, c] += sinus
simpan("contoh1_sinus.png", np.clip(sin_img, 0, 255))

for nama in ["gelap", "terang", "kontrasrendah", "saltpepper", "gaussian", "sinus"]:
    jawaban.append("contoh1_" + nama + ".png 5")

with open(os.path.join(FOLDER, "jawaban.txt"), "w") as f:
    f.write("\n".join(jawaban) + "\n")
print("Selesai, gambar disimpan di", FOLDER)
