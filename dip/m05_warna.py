# =============================================================================
# m05_warna.py  -  Materi 05 Color Image Processing
#   RGB -> HSI persis rumus slide:
#     I = (R + G + B) / 3
#     S = 1 - [3 / (R + G + B)] * min(R, G, B)
#     theta = cos^-1{ 1/2 [(R-G) + (R-B)] / [(R-G)^2 + (R-B)(G-B)]^(1/2) }
#     H = theta bila B <= G,  H = 360 - theta bila B > G
#   + jarak Euclid antar warna (De dari 02 Intro, dipakai di ruang RGB)
# =============================================================================
import math
import numpy as np

from dip.dasar import potong_nilai
from dip.m02_enhancement import histogram, persentil


def minimum_rgb(img):
    # min(R, G, B) per piksel
    mini = img[:, :, 0].copy()
    g = img[:, :, 1]
    b = img[:, :, 2]
    mini[g < mini] = g[g < mini]
    mini[b < mini] = b[b < mini]
    return mini


def rgb_ke_hsi(img):
    # img 0..255. Hasil: H (derajat 0..360), S (0..1), I (0..255)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    total = r + g + b

    intensitas = total / 3
    saturasi = 1 - 3 * minimum_rgb(img) / (total + 0.001)   # +0.001 agar tidak bagi 0

    atas = 0.5 * ((r - g) + (r - b))
    bawah = ((r - g) * (r - g) + (r - b) * (g - b)) ** 0.5
    cos_theta = potong_nilai(atas / (bawah + 0.001), -1, 1)
    # np.arccos = cos^-1 per elemen (sama dengan math.acos untuk satu angka)
    theta = np.arccos(cos_theta) * 180 / math.pi
    hue = theta.copy()
    biru_lebih = b > g
    hue[biru_lebih] = 360 - theta[biru_lebih]
    return hue, saturasi, intensitas


def jarak_warna(img, warna):
    # jarak Euclid De = sqrt((R-r)^2 + (G-g)^2 + (B-b)^2) ke satu warna
    return ((img[:, :, 0] - warna[0]) ** 2 + (img[:, :, 1] - warna[1]) ** 2 +
            (img[:, :, 2] - warna[2]) ** 2) ** 0.5


def jarak_dua_warna(w1, w2):
    return math.sqrt((w1[0] - w2[0]) ** 2 + (w1[1] - w2[1]) ** 2 + (w1[2] - w2[2]) ** 2)


def median_warna(img, mask):
    # median R, G, B dari piksel di dalam mask (dari histogram tiap kanal,
    # nilai tengah = persentil 50%)
    hasil = []
    for c in range(3):
        hist = histogram(img[:, :, c], mask)
        hasil.append(persentil(hist, 0.5))
    return hasil


def warna_rata(img, piksel):
    # rata-rata R, G, B dari daftar piksel (y, x)
    total = [0, 0, 0]
    for (y, x) in piksel:
        for c in range(3):
            total[c] += img[y, x, c]
    n = len(piksel)
    return [total[0] / n, total[1] / n, total[2] / n]


def hsi_satu_warna(warna):
    # rumus HSI yang sama, untuk satu warna [R, G, B]
    r = float(warna[0])
    g = float(warna[1])
    b = float(warna[2])
    total = r + g + b
    intensitas = total / 3
    saturasi = 1 - 3 * min(r, g, b) / (total + 0.001)
    atas = 0.5 * ((r - g) + (r - b))
    bawah = ((r - g) * (r - g) + (r - b) * (g - b)) ** 0.5
    cos_theta = atas / (bawah + 0.001)
    if cos_theta > 1:
        cos_theta = 1
    if cos_theta < -1:
        cos_theta = -1
    hue = math.acos(cos_theta) * 180 / math.pi
    if b > g:
        hue = 360 - hue
    return hue, saturasi, intensitas


def beda_hue(h1, h2):
    # hue berupa sudut: beda 350 dan 10 derajat = 20 derajat
    beda = abs(h1 - h2)
    if beda > 180:
        beda = 360 - beda
    return beda


def jarak_ke_abu(warna):
    # jarak Euclid warna ke sumbu abu-abu (I, I, I): 0 = tidak berwarna
    i = (warna[0] + warna[1] + warna[2]) / 3
    return math.sqrt((warna[0] - i) ** 2 + (warna[1] - i) ** 2 + (warna[2] - i) ** 2)
