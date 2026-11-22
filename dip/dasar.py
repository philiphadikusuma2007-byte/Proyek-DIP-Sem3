# =============================================================================
# dasar.py  -  Materi 01 & 02 Introduction to DIP
#   - citra digital = matriks f(x,y), 8 bit = 0..255, warna = 3 kanal RGB
#   - SAMPLING (digitisasi koordinat): memperkecil foto dengan mengambil
#     1 piksel setiap f piksel
#
# numpy hanya dipakai sebagai wadah angka (array) + operasi per elemen.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


def baca_gambar(path):
    # baca file -> array tinggi x lebar x 3, nilai 0..255
    img = plt.imread(path)
    if img.dtype != np.uint8:          # PNG dibaca 0..1 -> jadikan 0..255
        img = img * 255.0

    tinggi = img.shape[0]
    lebar = img.shape[1]
    hasil = np.zeros((tinggi, lebar, 3))
    if img.ndim == 2:                  # gambar abu-abu -> jadikan 3 kanal
        hasil[:, :, 0] = img
        hasil[:, :, 1] = img
        hasil[:, :, 2] = img
    else:                              # buang kanal alpha kalau ada
        hasil[:, :, 0] = img[:, :, 0]
        hasil[:, :, 1] = img[:, :, 1]
        hasil[:, :, 2] = img[:, :, 2]
    return hasil


def faktor_sampling(tinggi, lebar, maks_sisi):
    # ambil 1 piksel setiap f piksel supaya sisi terpanjang <= maks_sisi
    sisi = tinggi
    if lebar > sisi:
        sisi = lebar
    f = (sisi + maks_sisi - 1) // maks_sisi
    if f < 1:
        f = 1
    return f


def perkecil(img, maks_sisi):
    # SAMPLING: ambil 1 piksel setiap f piksel (tidak dirata-rata,
    # supaya noise salt & pepper tetap terlihat dan bisa dideteksi)
    f = faktor_sampling(img.shape[0], img.shape[1], maks_sisi)
    if f <= 1:
        return img.copy()
    return img[::f, ::f].copy()


def potong_nilai(a, bawah, atas):
    # clipping: nilai < bawah -> bawah, nilai > atas -> atas
    hasil = a.copy()
    hasil[hasil < bawah] = bawah
    hasil[hasil > atas] = atas
    return hasil


def ke_abu(img):
    # luminance (Y pada YIQ, materi 05): Y = 0.299 R + 0.587 G + 0.114 B
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def koordinat(tinggi, lebar):
    # yy[y, x] = y dan xx[y, x] = x  (dipakai untuk rumus jarak)
    yy = np.zeros((tinggi, lebar))
    xx = np.zeros((tinggi, lebar))
    for y in range(tinggi):
        yy[y, :] = y
    for x in range(lebar):
        xx[:, x] = x
    return yy, xx
