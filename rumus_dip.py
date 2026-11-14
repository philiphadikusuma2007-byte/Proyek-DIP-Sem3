# =============================================================================
# RUMUS DIP - semua rumus pengolahan citra ditulis MANUAL
# (tanpa cv2 / scipy / skimage / np.fft / np.histogram / np.median / dll.)
#
# numpy HANYA dipakai sebagai "wadah" angka (array) + operasi + - * / per
# elemen, supaya loop tidak terlalu lambat (tidak ngelag).
# Trik optimasi yang dipakai di seluruh file:
#   Daripada loop setiap piksel (y, x), yang di-loop adalah POSISI KERNEL.
#   Contoh filter 3x3: hanya 9 putaran, tiap putaran menjumlahkan satu
#   "gambar yang digeser". Rumusnya tetap sama persis dengan rumus di slide.
#
# Materi yang dipakai (sesuai slide):
#   02 Image Enhancement : histogram, contrast stretching, gamma,
#                          histogram equalization, thresholding (Otsu)
#   03 Spatial Filtering : konvolusi, lowpass (mean, gaussian), median,
#                          minimum / maksimum, Sobel
#   04 Frequency Domain  : DFT 2D, Butterworth (notch reject)
# =============================================================================

import math
import numpy as np
import matplotlib.pyplot as plt


# ============================ BACA & UKURAN ==================================
def baca_gambar(path):
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


def perkecil(img, maks_sisi):
    # Downsampling dengan SAMPLING: ambil 1 piksel setiap f piksel.
    # Foto HP 4000 px -> +-360 px, jadi semua proses jauh lebih cepat.
    # (Sengaja TIDAK dirata-rata: rata-rata akan menyembunyikan noise
    #  salt & pepper, sehingga cabang median filter tidak pernah terpilih.)
    tinggi = img.shape[0]
    lebar = img.shape[1]
    sisi = tinggi
    if lebar > sisi:
        sisi = lebar
    f = (sisi + maks_sisi - 1) // maks_sisi
    if f <= 1:
        return img.copy()
    return img[::f, ::f].copy()


def potong_nilai(a, bawah, atas):
    # clipping manual: nilai < bawah -> bawah, nilai > atas -> atas
    hasil = a.copy()
    hasil[hasil < bawah] = bawah
    hasil[hasil > atas] = atas
    return hasil


def ke_abu(img):
    # rumus luminance: 0.299 R + 0.587 G + 0.114 B
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


# ============================== HISTOGRAM ====================================
def histogram(abu):
    hist = [0] * 256
    nilai = potong_nilai(abu, 0, 255).astype(int)
    for v in nilai.ravel().tolist():
        hist[v] += 1
    return hist


def rata_dan_std(hist):
    n = 0
    total = 0
    for i in range(256):
        n += hist[i]
        total += i * hist[i]
    rata = total / n

    total_kuadrat = 0
    for i in range(256):
        total_kuadrat += (i - rata) * (i - rata) * hist[i]
    std = math.sqrt(total_kuadrat / n)
    return rata, std


def persentil(hist, p):
    # nilai keabuan di mana p% piksel berada di bawahnya
    n = 0
    for i in range(256):
        n += hist[i]
    kumulatif = 0
    for i in range(256):
        kumulatif += hist[i]
        if kumulatif >= p * n:
            return i
    return 255


def otsu(hist):
    # Thresholding Otsu: pilih T yang memaksimalkan variansi antar-kelas
    n = 0
    total = 0
    for i in range(256):
        n += hist[i]
        total += i * hist[i]

    terbaik = 0
    batas = 0
    n_bawah = 0
    total_bawah = 0
    for t in range(256):
        n_bawah += hist[t]
        if n_bawah == 0:
            continue
        n_atas = n - n_bawah
        if n_atas == 0:
            break
        total_bawah += t * hist[t]
        rata_bawah = total_bawah / n_bawah
        rata_atas = (total - total_bawah) / n_atas
        beda = rata_bawah - rata_atas
        variansi = n_bawah * n_atas * beda * beda
        if variansi > terbaik:
            terbaik = variansi
            batas = t
    return batas


def pakai_lut(img, lut):
    # lookup table: setiap nilai keabuan v diganti lut[v]
    tabel = np.array(lut)
    nilai = potong_nilai(img, 0, 255).astype(int)
    return tabel[nilai]


def lut_stretching(bawah, atas):
    # Contrast stretching: [bawah..atas] ditarik menjadi [0..255]
    lut = []
    for v in range(256):
        if atas <= bawah:
            baru = v
        else:
            baru = (v - bawah) * 255 / (atas - bawah)
        if baru < 0:
            baru = 0
        if baru > 255:
            baru = 255
        lut.append(baru)
    return lut


def lut_gamma(g):
    # Gamma correction: s = 255 * (r / 255) ^ g
    lut = []
    for v in range(256):
        lut.append(255 * math.pow(v / 255, g))
    return lut


def lut_equalization(hist):
    # Histogram equalization: s = 255 * CDF(r)
    n = 0
    for i in range(256):
        n += hist[i]
    lut = []
    kumulatif = 0
    for i in range(256):
        kumulatif += hist[i]
        lut.append(round(255 * kumulatif / n))
    return lut


# ============================ SPATIAL FILTER =================================
def pad_tepi(a, r):
    # perluas gambar r piksel di setiap sisi dengan mengulang piksel tepi
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
    # g(x,y) = sum_i sum_j h(i,j) * f(x+i, y+j)
    # yang di-loop posisi kernel (k x k), bukan setiap piksel
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
    kernel = []
    for i in range(k):
        kernel.append([1 / (k * k)] * k)
    return kernel


def kernel_gaussian(k, sigma):
    kernel = []
    total = 0
    r = k // 2
    for i in range(k):
        baris = []
        for j in range(k):
            w = math.exp(-((i - r) * (i - r) + (j - r) * (j - r)) / (2 * sigma * sigma))
            baris.append(w)
            total += w
        kernel.append(baris)
    for i in range(k):
        for j in range(k):
            kernel[i][j] = kernel[i][j] / total     # total bobot = 1
    return kernel


SOBEL_X = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1],
           [0, 0, 0],
           [1, 2, 1]]


def sobel(abu):
    gx = konvolusi(abu, SOBEL_X)
    gy = konvolusi(abu, SOBEL_Y)
    return (gx * gx + gy * gy) ** 0.5


def filter_median3(a):
    # Median 3x3: 9 tetangga diurutkan dengan bubble sort,
    # diambil nilai ke-5 (tengah). Cukup 5 putaran bubble sort,
    # karena setelah 5 putaran posisi ke-5 pasti sudah benar.
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


def per_kanal(img, fungsi, *argumen):
    # jalankan filter abu-abu pada kanal R, G, B satu per satu
    hasil = np.zeros(img.shape)
    for c in range(3):
        hasil[:, :, c] = fungsi(img[:, :, c], *argumen)
    return hasil


# ===================== FILTER MINIMUM / MAKSIMUM (MASK) ======================
# Untuk mask biner (0/1):
#   filter MINIMUM k x k = erosi  (objek menipis)
#   filter MAKSIMUM k x k = dilasi (objek menebal)
# Kernel kotak bisa dipisah: dulu arah horizontal, lalu vertikal.
def filter_minimum(mask, k):
    r = k // 2
    tinggi = mask.shape[0]
    lebar = mask.shape[1]
    p = pad_tepi(mask, r)
    sementara = np.ones((tinggi + 2 * r, lebar))
    for j in range(k):
        sementara = sementara * p[:, j:j + lebar]      # 0/1 dikali = minimum
    hasil = np.ones((tinggi, lebar))
    for i in range(k):
        hasil = hasil * sementara[i:i + tinggi, :]
    return hasil


def filter_maksimum(mask, k):
    # maksimum(mask) = 1 - minimum(1 - mask)
    return 1 - filter_minimum(1 - mask, k)


def opening(mask, k):
    return filter_maksimum(filter_minimum(mask, k), k)


def closing(mask, k):
    return filter_minimum(filter_maksimum(mask, k), k)


# ========================= PELABELAN REGION ==================================
def label_region(mask):
    # Region growing / flood fill (4 tetangga) memakai tumpukan (stack).
    # Hasil: peta label + daftar region (luas, kotak, daftar piksel).
    tinggi = mask.shape[0]
    lebar = mask.shape[1]
    data = (mask > 0.5).tolist()
    label = []
    for y in range(tinggi):
        label.append([0] * lebar)

    daftar = []
    nomor = 0
    for y in range(tinggi):
        for x in range(lebar):
            if data[y][x] and label[y][x] == 0:
                nomor += 1
                label[y][x] = nomor
                tumpukan = [(y, x)]
                piksel = []
                x0 = x
                x1 = x
                y0 = y
                y1 = y
                while len(tumpukan) > 0:
                    cy, cx = tumpukan.pop()
                    piksel.append((cy, cx))
                    if cx < x0:
                        x0 = cx
                    if cx > x1:
                        x1 = cx
                    if cy < y0:
                        y0 = cy
                    if cy > y1:
                        y1 = cy
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny >= 0 and ny < tinggi and nx >= 0 and nx < lebar:
                            if data[ny][nx] and label[ny][nx] == 0:
                                label[ny][nx] = nomor
                                tumpukan.append((ny, nx))
                daftar.append({"nomor": nomor, "luas": len(piksel), "piksel": piksel,
                               "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    return np.array(label), daftar


def isi_lubang(mask):
    # Lubang = daerah 0 yang TIDAK tersambung ke tepi gambar.
    tinggi = mask.shape[0]
    lebar = mask.shape[1]
    label, daftar = label_region(1 - mask)
    nyambung_tepi = [False] * (len(daftar) + 1)
    for x in range(lebar):
        nyambung_tepi[label[0, x]] = True
        nyambung_tepi[label[tinggi - 1, x]] = True
    for y in range(tinggi):
        nyambung_tepi[label[y, 0]] = True
        nyambung_tepi[label[y, lebar - 1]] = True

    hasil = mask.copy()
    for d in daftar:
        if not nyambung_tepi[d["nomor"]]:
            for (y, x) in d["piksel"]:
                hasil[y, x] = 1
    return hasil


def region_terbesar(mask):
    label, daftar = label_region(mask)
    hasil = np.zeros(mask.shape)
    if len(daftar) == 0:
        return hasil
    besar = daftar[0]
    for d in daftar:
        if d["luas"] > besar["luas"]:
            besar = d
    for (y, x) in besar["piksel"]:
        hasil[y, x] = 1
    return hasil


# ======================= DOMAIN FREKUENSI (DFT) ==============================
def dft_tiap_baris(a, tanda):
    # DFT 1D pada setiap baris:  F(u) = sum_x f(x) * e^(tanda * j*2*pi*u*x/N)
    # tanda = -1 untuk DFT, +1 untuk invers.
    # Semua baris dihitung bersamaan (yang di-loop hanya u dan x).
    tinggi = a.shape[0]
    n = a.shape[1]
    kolom = a.T.copy()                   # kolom[x] = nilai kolom ke-x (semua baris)
    hasil = np.zeros((n, tinggi), dtype=complex)
    for u in range(n):
        jumlah = np.zeros(tinggi, dtype=complex)
        for x in range(n):
            sudut = 2 * math.pi * ((u * x) % n) / n
            jumlah += kolom[x] * complex(math.cos(sudut), tanda * math.sin(sudut))
        hasil[u] = jumlah
    return hasil.T


def dft2(a):
    # DFT 2D separabel: DFT per baris, lalu DFT per kolom
    F = dft_tiap_baris(a, -1)
    F = dft_tiap_baris(F.T, -1).T
    return F


def idft2(F):
    tinggi = F.shape[0]
    lebar = F.shape[1]
    f = dft_tiap_baris(F, 1)
    f = dft_tiap_baris(f.T, 1).T
    return f / (tinggi * lebar)


def geser_tengah(F):
    # pindahkan frekuensi (0,0) ke tengah spektrum (tukar 4 kuadran)
    tinggi = F.shape[0]
    lebar = F.shape[1]
    hasil = np.zeros(F.shape, dtype=F.dtype)
    for y in range(tinggi):
        hasil[(y + tinggi // 2) % tinggi] = np.concatenate(
            (F[y, lebar - lebar // 2:], F[y, :lebar - lebar // 2]))
    return hasil


def butterworth_notch_reject(tinggi, lebar, daftar_puncak, d0, orde):
    # H = produk dari [1 / (1 + (D0 / Dk)^(2n))] untuk setiap puncak dan
    # pasangan simetrisnya (spektrum sudah digeser ke tengah).
    yy = np.zeros((tinggi, lebar))
    xx = np.zeros((tinggi, lebar))
    for y in range(tinggi):
        yy[y, :] = y
    for x in range(lebar):
        xx[:, x] = x

    H = np.ones((tinggi, lebar))
    cy = tinggi // 2
    cx = lebar // 2
    for (py, px) in daftar_puncak:
        for (ky, kx) in ((py, px), (2 * cy - py, 2 * cx - px)):
            d = ((yy - ky) ** 2 + (xx - kx) ** 2) ** 0.5
            d[d < 0.001] = 0.001
            H = H * (1 / (1 + (d0 / d) ** (2 * orde)))
    return H


# ======================= CIRI LOKAL & DESKRIPTOR REGION ======================
def std_lokal(abu, k):
    # simpangan baku di jendela k x k:  std = sqrt( mean(f^2) - mean(f)^2 )
    # kecil = daerah HALUS (obat), besar = daerah KASAR (foil bertekstur)
    rata = konvolusi(abu, kernel_mean(k))
    rata_kuadrat = konvolusi(abu * abu, kernel_mean(k))
    variansi = rata_kuadrat - rata * rata
    variansi[variansi < 0] = 0
    return variansi ** 0.5


def saturasi(img):
    # S = (maks(R,G,B) - min(R,G,B)) / maks   (0 = abu-abu, 1 = warna penuh)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    maks = r.copy()
    maks[g > maks] = g[g > maks]
    maks[b > maks] = b[b > maks]
    mini = r.copy()
    mini[g < mini] = g[g < mini]
    mini[b < mini] = b[b < mini]
    return (maks - mini) / (maks + 1)


def jarak_ke_warna(img, warna):
    # jarak Euclid setiap piksel ke satu warna [R, G, B]
    return ((img[:, :, 0] - warna[0]) ** 2 + (img[:, :, 1] - warna[1]) ** 2 +
            (img[:, :, 2] - warna[2]) ** 2) ** 0.5


def median_warna(img, mask):
    # median R, G, B dari piksel mask (lewat histogram, tanpa np.median)
    hasil = []
    for c in range(3):
        hist = [0] * 256
        nilai = potong_nilai(img[:, :, c], 0, 255).astype(int)
        for v in nilai[mask > 0.5].tolist():
            hist[v] += 1
        hasil.append(persentil(hist, 0.5))
    return hasil


def momen_region(piksel):
    # Momen pusat orde 2 -> elips yang setara dengan region.
    #   mu20, mu02, mu11 -> lambda1, lambda2 (sumbu panjang & pendek)
    #   lonjong = sqrt(lambda1 / lambda2)          (tidak terpengaruh rotasi)
    #   isi     = luas / luas elips (4 pi sqrt(l1 l2))  ~1 untuk obat utuh,
    #             kecil untuk bentuk aneh (huruf L, U, cabang-cabang)
    n = len(piksel)
    sy = 0
    sx = 0
    for (y, x) in piksel:
        sy += y
        sx += x
    cy = sy / n
    cx = sx / n
    m20 = 0
    m02 = 0
    m11 = 0
    for (y, x) in piksel:
        m20 += (y - cy) * (y - cy)
        m02 += (x - cx) * (x - cx)
        m11 += (y - cy) * (x - cx)
    m20 = m20 / n
    m02 = m02 / n
    m11 = m11 / n
    tengah = (m20 + m02) / 2
    akar = math.sqrt(((m20 - m02) / 2) ** 2 + m11 * m11)
    l1 = tengah + akar
    l2 = tengah - akar
    if l2 < 0.01:
        l2 = 0.01
    isi = n / (4 * math.pi * math.sqrt(l1 * l2))
    lonjong = math.sqrt(l1 / l2)
    return cy, cx, isi, lonjong


def kontur(mask):
    # garis tepi region = mask - erosi(mask)
    return mask - filter_minimum(mask, 3)
