# =============================================================================
# m03_spasial.py  -  Materi 03 Spatial Filtering
#   - konvolusi 3x3 / kxk:  g(m,n) = sum w_i f(...)  (input selalu dari
#     citra asli, tepi = replikasi piksel tepi, sesuai aturan slide)
#   - lowpass: (1/9)[1 1 1;1 1 1;1 1 1] dan (1/32)[1 3 1;3 16 3;1 3 1]
#   - median filter (non-linear)
#   - filter minimum / maksimum (non-linear)
#   - mask Sobel h1, h2 (dipakai di m08 untuk deteksi tepi)
#
# Trik kecepatan: yang di-loop adalah POSISI KERNEL (9 putaran untuk 3x3),
# setiap putaran menjumlahkan satu "gambar yang digeser". Rumusnya sama.
# =============================================================================
import numpy as np


# mask lowpass dari slide (total bobot = 1)
LOWPASS_32 = [[1 / 32, 3 / 32, 1 / 32],
              [3 / 32, 16 / 32, 3 / 32],
              [1 / 32, 3 / 32, 1 / 32]]

# mask Sobel dari slide
SOBEL_H1 = [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]
SOBEL_H2 = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]


def pad_tepi(a, r):
    # perluas gambar r piksel di tiap sisi dengan mengulang piksel tepi
    tinggi = a.shape[0]
    lebar = a.shape[1]
    p = np.zeros((tinggi + 2 * r, lebar + 2 * r))
    p[r:r + tinggi, r:r + lebar] = a
    for i in range(r):
        p[i, r:r + lebar] = a[0]
        p[r + tinggi + i, r:r + lebar] = a[tinggi - 1]
    for j in range(r):
        p[:, j] = p[:, r]
        p[:, r + lebar + j] = p[:, r + lebar - 1]
    return p


def konvolusi(a, kernel):
    # g(m,n) = sum_i sum_j w(i,j) * f(m+i-r, n+j-r)
    k = len(kernel)
    r = k // 2
    tinggi = a.shape[0]
    lebar = a.shape[1]
    p = pad_tepi(a, r)
    hasil = np.zeros((tinggi, lebar))
    for i in range(k):
        for j in range(k):
            if kernel[i][j] != 0:
                hasil += kernel[i][j] * p[i:i + tinggi, j:j + lebar]
    return hasil


def kernel_mean(k):
    # (1/N^2) * ones(N x N)  -> lowpass rata-rata
    kernel = []
    for i in range(k):
        kernel.append([1 / (k * k)] * k)
    return kernel


def filter_median3(a):
    # Median 3x3: 9 tetangga diurutkan (bubble sort), ambil urutan ke-5.
    # Cukup 5 putaran bubble sort: setelah itu posisi ke-5 sudah benar.
    tinggi = a.shape[0]
    lebar = a.shape[1]
    p = pad_tepi(a, 1)
    daftar = []
    for i in range(3):
        for j in range(3):
            daftar.append(p[i:i + tinggi, j:j + lebar].copy())
    for putaran in range(5):
        for k in range(8 - putaran):
            kiri = daftar[k]
            kanan = daftar[k + 1]
            tukar = kiri > kanan
            simpan = kiri[tukar]
            kiri[tukar] = kanan[tukar]
            kanan[tukar] = simpan
    return daftar[4]


def per_kanal(img, fungsi, argumen=None):
    # jalankan filter abu-abu pada kanal R, G, B satu per satu
    hasil = np.zeros(img.shape)
    for c in range(3):
        if argumen is None:
            hasil[:, :, c] = fungsi(img[:, :, c])
        else:
            hasil[:, :, c] = fungsi(img[:, :, c], argumen)
    return hasil


def filter_minimum(a, k):
    # Filter MINIMUM k x k (untuk mask 0/1). Kotak k x k bisa dipisah:
    # dulu arah horizontal, lalu vertikal. Pada mask 0/1, perkalian = minimum.
    r = k // 2
    tinggi = a.shape[0]
    lebar = a.shape[1]
    p = pad_tepi(a, r)
    sementara = np.ones((tinggi + 2 * r, lebar))
    for j in range(k):
        sementara = sementara * p[:, j:j + lebar]
    hasil = np.ones((tinggi, lebar))
    for i in range(k):
        hasil = hasil * sementara[i:i + tinggi, :]
    return hasil


def filter_maksimum(a, k):
    # Filter MAKSIMUM k x k untuk mask 0/1:  maks(a) = 1 - min(1 - a)
    return 1 - filter_minimum(1 - a, k)


def std_lokal(abu, k):
    # simpangan baku di jendela k x k, dari dua lowpass rata-rata:
    #   std = sqrt( mean(f^2) - mean(f)^2 )
    # kecil = daerah HALUS (obat), besar = daerah KASAR (foil bertekstur)
    rata = konvolusi(abu, kernel_mean(k))
    rata_kuadrat = konvolusi(abu * abu, kernel_mean(k))
    variansi = rata_kuadrat - rata * rata
    variansi[variansi < 0] = 0
    return variansi ** 0.5
