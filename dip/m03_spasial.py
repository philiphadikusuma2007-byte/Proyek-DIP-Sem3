# =============================================================================
# dip/m03_spasial.py  -  Materi 03 Spatial Filtering
#
#   Sliding window: filter n x n digeser sampai pusatnya mengenai semua piksel.
#   Aturan slide yang diikuti:
#     - hasil ditulis ke CITRA BARU, input selalu dari citra asli
#     - di pinggir citra: "perlebar citra, piksel di pinggir diisi perulangan
#       nilai piksel pada pinggir tersebut" (replikasi tepi)
#
#   Linear  : konvolusi 3x3  g(m,n) = sum w_i f(...);  lowpass rata-rata
#             (1/N^2)[1 ... 1];  lowpass (1/32)[1 3 1; 3 16 3; 1 3 1];  Sobel
#   Non-linear: median 3x3, filter MINIMUM dan MAKSIMUM n x n
#
#   Trik kecepatan (rumus tetap sama):
#     * Filter rata-rata N x N = jumlah N x N / N^2. Jumlah jendela dihitung
#       dengan "jumlah berjalan": jendela digeser 1 piksel -> tambah nilai yang
#       masuk, kurangi nilai yang keluar. Kotak N x N bisa dipisah menjadi
#       arah horizontal lalu vertikal (separable).
#     * Filter minimum/maksimum N x N juga separable (min kotak = min baris
#       dari min kolom) dan dihitung dengan algoritma van Herk / Gil-Werman:
#       setiap baris dibagi blok N piksel, dihitung min kumulatif dari kiri
#       dan dari kanan di tiap blok, lalu min jendela = min(kanan[i], kiri[i+N-1]).
#
# File ini TIDAK meng-import library apa pun (hanya modul dip sendiri).
# =============================================================================
from dip.dasar import akar, transpos

# mask lowpass dari slide 03 (total bobot = 1)
LOWPASS_32 = [[1, 3, 1],
              [3, 16, 3],
              [1, 3, 1]]          # dikali 1/32

# mask Sobel dari slide 03 / slide 08
SOBEL_H1 = [[-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]]            # turunan arah vertikal   (Gx di slide 08)
SOBEL_H2 = [[-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]]           # turunan arah horizontal (Gy di slide 08)


# ----------------------------------------------------------------------------
# KONVOLUSI 3x3 (rumus slide 03)
# ----------------------------------------------------------------------------
def konvolusi3(data, lebar, tinggi, kernel, pembagi=1):
    # g(m,n) = w1 f(m-1,n-1) + w2 f(m-1,n) + w3 f(m-1,n+1)
    #        + w4 f(m,n-1)   + w5 f(m,n)   + w6 f(m,n+1)
    #        + w7 f(m+1,n-1) + w8 f(m+1,n) + w9 f(m+1,n+1)
    # Setiap baris diperlebar 1 piksel kiri-kanan (replikasi tepi), baris
    # pertama/terakhir diulang untuk tepi atas/bawah.
    (w1, w2, w3), (w4, w5, w6), (w7, w8, w9) = kernel
    baris = []
    for y in range(tinggi):
        b = data[y * lebar:(y + 1) * lebar]
        baris.append([b[0]] + b + [b[-1]])
    hasil = []
    for y in range(tinggi):
        a = baris[y - 1] if y > 0 else baris[0]            # baris atas
        t = baris[y]                                       # baris tengah
        c = baris[y + 1] if y < tinggi - 1 else baris[-1]  # baris bawah
        # a[0:], a[1:], a[2:] = tetangga kiri, tengah, kanan
        hasil.extend([(w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4 + w5 * p5 + w6 * p6 +
                       w7 * p7 + w8 * p8 + w9 * p9) / pembagi
                      for p1, p2, p3, p4, p5, p6, p7, p8, p9 in
                      zip(a, a[1:], a[2:], t, t[1:], t[2:], c, c[1:], c[2:])])
    return hasil


def sobel(abu, lebar, tinggi):
    # Deteksi tepi Sobel (slide 03 & 08):  grad f = |Gx| + |Gy|
    gx = konvolusi3(abu, lebar, tinggi, SOBEL_H1)
    gy = konvolusi3(abu, lebar, tinggi, SOBEL_H2)
    return [(p if p > 0 else -p) + (q if q > 0 else -q) for p, q in zip(gx, gy)]


# ----------------------------------------------------------------------------
# FILTER RATA-RATA N x N  (lowpass 1/N^2 [1 ... 1], slide 03)
# ----------------------------------------------------------------------------
def _jumlah_baris(data, lebar, tinggi, r):
    # jumlah jendela horizontal [x-r .. x+r] untuk setiap piksel,
    # dengan replikasi tepi, memakai jumlah berjalan
    hasil = []
    for y in range(tinggi):
        b = data[y * lebar:(y + 1) * lebar]
        p = [b[0]] * r + b + [b[-1]] * r           # perlebar r piksel kiri-kanan
        s = 0
        for v in p[:2 * r + 1]:
            s += v
        hasil.append(s)
        keluar = 0                                 # indeks nilai yang keluar jendela
        for masuk in p[2 * r + 1:]:
            s += masuk - p[keluar]
            keluar += 1
            hasil.append(s)
    return hasil


def jumlah_kotak(data, lebar, tinggi, k):
    # jumlah semua nilai di jendela k x k (k ganjil) di sekitar setiap piksel.
    # separable: jumlah baris dulu, lalu jumlah kolom (pakai transpos).
    r = k // 2
    h = _jumlah_baris(data, lebar, tinggi, r)
    h = transpos(h, lebar, tinggi)                 # kolom menjadi baris
    v = _jumlah_baris(h, tinggi, lebar, r)
    return transpos(v, tinggi, lebar)


def rata_kotak(data, lebar, tinggi, k):
    # lowpass rata-rata k x k:  g = (1/k^2) * jumlah k x k
    n = float(k * k)
    return [s / n for s in jumlah_kotak(data, lebar, tinggi, k)]


def std_lokal(data, lebar, tinggi, k):
    # simpangan baku di jendela k x k, dari dua lowpass rata-rata:
    #   std = sqrt( mean(f^2) - mean(f)^2 )
    # kecil = permukaan HALUS (obat), besar = permukaan KASAR (foil bertekstur)
    m = rata_kotak(data, lebar, tinggi, k)
    m2 = rata_kotak([v * v for v in data], lebar, tinggi, k)
    return [akar(b - a * a) for a, b in zip(m, m2)]


# ----------------------------------------------------------------------------
# MEDIAN 3x3 (non-linear, slide 03; order-statistic filter, slide 06)
# ----------------------------------------------------------------------------
def median3(data, lebar, tinggi):
    # 9 nilai di sekitar piksel diurutkan, diambil urutan ke-5 (nilai tengah)
    baris = []
    for y in range(tinggi):
        b = data[y * lebar:(y + 1) * lebar]
        baris.append([b[0]] + b + [b[-1]])
    hasil = []
    for y in range(tinggi):
        a = baris[y - 1] if y > 0 else baris[0]
        t = baris[y]
        c = baris[y + 1] if y < tinggi - 1 else baris[-1]
        hasil.extend([sorted(n)[4] for n in
                      zip(a, a[1:], a[2:], t, t[1:], t[2:], c, c[1:], c[2:])])
    return hasil


# ----------------------------------------------------------------------------
# FILTER MINIMUM / MAKSIMUM N x N (non-linear, slide 03)
# ----------------------------------------------------------------------------
def _min_baris(data, lebar, tinggi, r):
    # minimum jendela horizontal [x-r .. x+r] (van Herk / Gil-Werman)
    k = 2 * r + 1
    hasil = []
    for y in range(tinggi):
        b = data[y * lebar:(y + 1) * lebar]
        p = [b[0]] * r + b + [b[-1]] * r
        m = len(p)
        kiri = [0] * m                     # min kumulatif dari awal blok
        kanan = [0] * m                    # min kumulatif dari akhir blok
        for i in range(m):
            v = p[i]
            if i % k == 0 or v < kiri[i - 1]:
                kiri[i] = v
            else:
                kiri[i] = kiri[i - 1]
        for i in range(m - 1, -1, -1):
            v = p[i]
            if i == m - 1 or (i + 1) % k == 0 or v < kanan[i + 1]:
                kanan[i] = v
            else:
                kanan[i] = kanan[i + 1]
        # jendela p[i .. i+k-1] = baris asli [x-r .. x+r]
        hasil.extend([u if u < w else w for u, w in zip(kanan, kiri[k - 1:])])
    return hasil


def filter_minimum(data, lebar, tinggi, k):
    # g(x,y) = nilai TERKECIL di jendela k x k  (= erosi graylevel, SE kotak)
    r = k // 2
    if r == 0:
        return list(data)
    h = _min_baris(data, lebar, tinggi, r)
    h = transpos(h, lebar, tinggi)
    v = _min_baris(h, tinggi, lebar, r)
    return transpos(v, tinggi, lebar)


def filter_maksimum(data, lebar, tinggi, k):
    # g(x,y) = nilai TERBESAR di jendela k x k  (= dilasi graylevel, SE kotak)
    # max(f) = -min(-f), jadi cukup memakai filter minimum
    neg = [-v for v in data]
    return [-v for v in filter_minimum(neg, lebar, tinggi, k)]
