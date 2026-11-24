# =============================================================================
# dip/m02_enhancement.py  -  Materi 02 Image Enhancement (Point Operation)
#
#   Point operation:  s = T(r)   (output hanya bergantung pada 1 piksel)
#   - Histogram               p(r_k) = n_k / n
#   - Contrast stretching     garis lurus (r1, 0) -> (r2, 255)
#   - Thresholding            kasus khusus contrast stretching (r1 = r2)
#   - Gamma correction        T(r) = ((r - r1) / (r2 - r1)) ^ g   (versi slide)
#   - Histogram equalization  s_k = sum_{j=0..k} p(r_j)
#   - Graylevel slicing       tonjolkan rentang graylevel tertentu
#
# Semua operasi titik dikerjakan lewat LUT (look-up table): tabel 256 nilai
# T(0), T(1), ..., T(255) dihitung sekali, lalu setiap piksel r diganti lut[r].
#
# File ini TIDAK meng-import library apa pun.
# =============================================================================
from dip.dasar import akar


def histogram(data, mask=None):
    # n_k = banyak piksel bernilai k (k = 0..255). data harus int 0..255.
    # Bila mask diberikan, hanya piksel dengan mask = 1 yang dihitung.
    hist = [0] * 256
    if mask is None:
        for v in data:
            hist[v] += 1
    else:
        for v, m in zip(data, mask):
            if m:
                hist[v] += 1
    return hist


def rata_dan_std(hist):
    # rata-rata  mu = sum k p(k);   simpangan baku sigma = sqrt(sum (k-mu)^2 p(k))
    n = 0
    total = 0
    for k in range(256):
        n += hist[k]
        total += k * hist[k]
    if n == 0:
        return 0.0, 0.0
    mu = total / n
    var = 0.0
    for k in range(256):
        var += (k - mu) * (k - mu) * hist[k]
    return mu, akar(var / n)


def persentil(hist, bagian):
    # nilai k terkecil yang di bawahnya sudah ada (bagian x 100)% piksel
    # (dari histogram kumulatif). bagian = 0.5 -> median.
    n = 0
    for k in range(256):
        n += hist[k]
    kumulatif = 0
    for k in range(256):
        kumulatif += hist[k]
        if kumulatif >= bagian * n:
            return k
    return 255


def pakai_lut(data, lut):
    # point operation s = T(r): setiap piksel r diganti lut[r]
    return [lut[v] for v in data]


def lut_stretching(r1, r2):
    # Contrast stretching (slide 02):
    #   r <= r1 -> 0,  r >= r2 -> 255,  di antaranya garis lurus
    #   s = (r - r1) * 255 / (r2 - r1)
    lut = []
    for r in range(256):
        if r2 <= r1:
            s = r
        else:
            s = (r - r1) * 255.0 / (r2 - r1)
        if s < 0:
            s = 0
        if s > 255:
            s = 255
        lut.append(int(s + 0.5))
    return lut


def lut_gamma(r1, r2, g):
    # Gamma correction versi slide 02:
    #   T(r) = 0 (r < r1);  ((r - r1)/(r2 - r1))^g (r1 <= r <= r2);  1 (r > r2)
    # g < 1 -> lebih terang, g > 1 -> lebih gelap. Dikali 255 (8 bit).
    lut = []
    for r in range(256):
        if r < r1:
            s = 0.0
        elif r > r2:
            s = 1.0
        else:
            s = ((r - r1) / (r2 - r1)) ** g
        lut.append(int(255 * s + 0.5))
    return lut


def lut_equalization(hist):
    # Histogram equalization (slide 02):
    #   s_k = T(r_k) = sum_{j=0..k} p(r_j)   lalu dikali 255 dan dibulatkan
    n = 0
    for k in range(256):
        n += hist[k]
    lut = []
    kumulatif = 0
    for k in range(256):
        kumulatif += hist[k]
        lut.append(int(255.0 * kumulatif / n + 0.5))
    return lut


def threshold(data, t):
    # Thresholding / binarisasi: 1 bila r > T, 0 bila tidak
    return [1 if v > t else 0 for v in data]


def graylevel_slicing(data, bawah, atas):
    # Graylevel slicing TANPA background: rentang [bawah..atas] -> 1, lainnya 0
    return [1 if bawah <= v <= atas else 0 for v in data]


def skala_8bit(data, bawah, atas):
    # skala linear [bawah..atas] -> [0..255] (int), di luar rentang dijepit.
    # Dipakai agar data apa pun (jarak warna, top-hat, ...) bisa dibuat
    # histogram 256 level.
    if atas <= bawah:
        atas = bawah + 1
    k = 255.0 / (atas - bawah)
    hasil = []
    for v in data:
        s = (v - bawah) * k
        if s < 0:
            hasil.append(0)
        elif s > 255:
            hasil.append(255)
        else:
            hasil.append(int(s + 0.5))
    return hasil
