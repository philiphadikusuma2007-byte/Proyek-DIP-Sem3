# =============================================================================
# m02_enhancement.py  -  Materi 02 Image Enhancement (point operation)
#   - histogram          p(r_k) = n_k / n
#   - histogram equalization   s_k = sum_{j=0..k} p(r_j)
#   - contrast stretching      garis lurus (r1,0) -> (r2,255)
#   - gamma (versi slide)      T(r) = ((r - r1) / (r2 - r1)) ^ g
#   - thresholding / binarisasi  (contrast stretching dengan r1 = r2)
#   - graylevel slicing        tonjolkan satu rentang nilai
# =============================================================================
import math
import numpy as np

from dip.dasar import potong_nilai


def histogram(abu, mask=None):
    # n_k = banyak piksel bernilai k (0..255). Bila mask diberikan,
    # hanya piksel di dalam mask yang dihitung.
    hist = [0] * 256
    nilai = potong_nilai(abu, 0, 255).astype(int)
    if mask is None:
        daftar = nilai.ravel().tolist()
    else:
        daftar = nilai[mask > 0.5].tolist()
    for v in daftar:
        hist[v] += 1
    return hist


def probabilitas(hist):
    # p(r_k) = n_k / n
    n = 0
    for i in range(256):
        n += hist[i]
    p = []
    for i in range(256):
        p.append(hist[i] / n)
    return p


def rata_dan_std(hist):
    # mu = sum k p(k),  sigma^2 = sum (k - mu)^2 p(k)
    p = probabilitas(hist)
    rata = 0
    for i in range(256):
        rata += i * p[i]
    variansi = 0
    for i in range(256):
        variansi += (i - rata) * (i - rata) * p[i]
    return rata, math.sqrt(variansi)


def persentil(hist, bagian):
    # nilai keabuan k di mana (bagian x 100)% piksel berada di bawahnya
    # (dihitung dari jumlah kumulatif histogram). bagian=0.5 -> median.
    n = 0
    for i in range(256):
        n += hist[i]
    kumulatif = 0
    for i in range(256):
        kumulatif += hist[i]
        if kumulatif >= bagian * n:
            return i
    return 255


def pakai_lut(img, lut):
    # point operation s = T(r): setiap nilai r diganti lut[r]
    tabel = np.array(lut)
    nilai = potong_nilai(img, 0, 255).astype(int)
    return tabel[nilai]


def lut_stretching(r1, r2):
    # Contrast stretching: r <= r1 -> 0, r >= r2 -> 255, di antaranya garis lurus
    lut = []
    for r in range(256):
        if r2 <= r1:
            s = r
        else:
            s = (r - r1) * 255 / (r2 - r1)
        if s < 0:
            s = 0
        if s > 255:
            s = 255
        lut.append(s)
    return lut


def lut_gamma(r1, r2, g):
    # Gamma versi slide: T(r) = 0 (r < r1), ((r-r1)/(r2-r1))^g, 1 (r > r2)
    # g < 1 -> lebih terang, g > 1 -> lebih gelap. Dikali 255 (8 bit).
    lut = []
    for r in range(256):
        if r < r1:
            s = 0
        elif r > r2:
            s = 1
        else:
            s = math.pow((r - r1) / (r2 - r1), g)
        lut.append(255 * s)
    return lut


def lut_equalization(hist):
    # Histogram equalization: s_k = sum_{j=0..k} p(r_j), lalu x 255 dibulatkan
    p = probabilitas(hist)
    lut = []
    kumulatif = 0
    for k in range(256):
        kumulatif += p[k]
        lut.append(round(255 * kumulatif))
    return lut


def threshold(gambar, t):
    # binarisasi: 1 bila nilai > t, 0 bila tidak
    return (gambar > t) * 1.0


def graylevel_slicing(gambar, bawah, atas):
    # slicing tanpa background: rentang [bawah..atas] -> 1, lainnya -> 0
    return (gambar >= bawah) * (gambar <= atas) * 1.0


def ke_8bit(a, maks):
    # skala linear 0..maks -> 0..255 (supaya bisa dibuat histogram 256 level)
    return potong_nilai(a * 255 / (maks + 0.001), 0, 255)
