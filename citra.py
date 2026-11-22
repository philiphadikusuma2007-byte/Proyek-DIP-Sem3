# =============================================================================
# citra.py  -  KUMPULAN RUMUS PENGOLAHAN CITRA DIGITAL (DITULIS MANUAL)
#
# ATURAN FILE INI:
#   * TIDAK memakai library apa pun (tidak ada numpy, cv2, PIL, scipy,
#     bahkan modul "math" pun tidak dipakai).
#   * Hanya memakai Python dasar: list, for, if, + - * / ** (pangkat).
#     Akar kuadrat ditulis sebagai x ** 0.5.
#
# Cara menyimpan citra:
#   Citra 1 kanal disimpan sebagai LIST DATAR (flat list) berukuran
#   lebar x tinggi. Piksel (x, y) ada di indeks  i = y * lebar + x.
#   Citra berwarna = 3 list datar: R, G, B (nilai 0..255).
#   Mask biner = list datar berisi 0 / 1.
#
# Materi yang dipakai (PPT kuliah):
#   01-02 Introduction      : sampling, kuantisasi, ketetanggaan piksel
#   02 Image Enhancement    : histogram, statistik, contrast stretching,
#                             gamma, histogram equalization, thresholding
#   03 Spatial Filtering    : sliding window, lowpass (mean), median,
#                             minimum/maksimum, perlebar tepi (replicate)
#   05 Color Processing     : RGB -> grayscale, YCbCr (Cb), HSI (saturasi),
#                             HSV (V), normalized RGB (kromatisitas)
#   08 Image Segmentation   : threshold Otsu, region growing,
#                             connected component, highpass (energi tekstur)
#   09 Representation       : dilasi, erosi, opening, closing, isi lubang,
#                             kode rantai (chain code), keliling, momen
#   06 Restoration          : order statistic (median) untuk estimasi latar
# =============================================================================


# ============================ 1. SAMPLING ====================================
def sampling_ulang(data_rgb, lebar_asli, tinggi_asli, kotak, maks_sisi, rata=True):
    """Memotong (crop) lalu memperkecil foto secara manual.

    data_rgb : bytes R,G,B,R,G,B,... dari foto asli (urut baris).
    kotak    : (x0, y0, x1, y1) daerah yang diambil dari foto asli.
    maks_sisi: sisi terpanjang hasil.

    Rumus (sampling + rata-rata lokal / anti-aliasing):
      faktor f = sisi_terpanjang_kotak / maks_sisi
      piksel hasil (x, y) = rata-rata 3x3 titik sampel di dalam blok
      asli berukuran f x f yang diwakilinya.
    Dirata-rata (bukan hanya diambil 1 titik) supaya pola halus foil tidak
    berubah jadi pola palsu (aliasing).
    """
    x0, y0, x1, y1 = kotak
    lebar_kotak = x1 - x0
    tinggi_kotak = y1 - y0
    sisi = lebar_kotak if lebar_kotak > tinggi_kotak else tinggi_kotak
    f = sisi / maks_sisi
    if f < 1:
        f = 1.0
    lebar = int(lebar_kotak / f)
    tinggi = int(tinggi_kotak / f)

    R = [0.0] * (lebar * tinggi)
    G = [0.0] * (lebar * tinggi)
    B = [0.0] * (lebar * tinggi)
    # posisi 3 titik sampel di dalam satu blok: 1/6, 3/6, 5/6 dari lebar blok
    # (rata=False: hanya 1 titik di tengah blok = sampling murni)
    geser = [f / 6, f / 2, 5 * f / 6] if rata else [f / 2]
    n_sampel = len(geser) * len(geser)
    for y in range(tinggi):
        baris_asal = []
        for gy in geser:
            ya = int(y0 + y * f + gy)
            if ya > tinggi_asli - 1:
                ya = tinggi_asli - 1
            baris_asal.append(ya * lebar_asli)
        for x in range(lebar):
            jr = 0
            jg = 0
            jb = 0
            for gx in geser:
                xa = int(x0 + x * f + gx)
                if xa > lebar_asli - 1:
                    xa = lebar_asli - 1
                for awal in baris_asal:
                    p = (awal + xa) * 3
                    jr += data_rgb[p]
                    jg += data_rgb[p + 1]
                    jb += data_rgb[p + 2]
            i = y * lebar + x
            R[i] = jr / n_sampel
            G[i] = jg / n_sampel
            B[i] = jb / n_sampel
    return R, G, B, lebar, tinggi, f


# ======================= 2. WARNA (05 COLOR PROCESSING) ======================
def ke_abu(R, G, B):
    # Grayscale (luminance):  Y = 0.299 R + 0.587 G + 0.114 B
    return [0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i] for i in range(len(R))]


def ke_cb(R, G, B):
    # Komponen krominansi biru dari model YCbCr (keluarga YUV/YIQ):
    #   Cb = 128 - 0.169 R - 0.331 G + 0.5 B
    # Cb > 128 -> kebiruan (foil perak memantulkan cahaya kebiruan)
    # Cb < 128 -> kekuningan / kecokelatan (meja kayu, obat krem)
    return [128 - 0.169 * R[i] - 0.331 * G[i] + 0.5 * B[i] for i in range(len(R))]


def saturasi_hsi(R, G, B):
    # Model HSI:  S = 1 - 3 * min(R, G, B) / (R + G + B)
    #   S = 0 -> abu-abu/putih/hitam (foil perak),  S besar -> berwarna
    hasil = [0.0] * len(R)
    for i in range(len(R)):
        r = R[i]
        g = G[i]
        b = B[i]
        terkecil = r
        if g < terkecil:
            terkecil = g
        if b < terkecil:
            terkecil = b
        total = r + g + b
        if total < 1:
            hasil[i] = 0.0
        else:
            hasil[i] = 1 - 3 * terkecil / total
    return hasil


def jarak_warna(R, G, B, warna):
    # Jarak Euclid di ruang RGB:  d = sqrt((R-r)^2 + (G-g)^2 + (B-b)^2)
    r0, g0, b0 = warna
    return [((R[i] - r0) ** 2 + (G[i] - g0) ** 2 + (B[i] - b0) ** 2) ** 0.5
            for i in range(len(R))]


def warna_median(R, G, B, mask):
    # median tiap kanal untuk piksel yang mask-nya 1 (lewat histogram)
    hasil = []
    for kanal in (R, G, B):
        hist = [0] * 256
        for i in range(len(kanal)):
            if mask[i]:
                hist[potong(kanal[i])] += 1
        hasil.append(persentil(hist, 0.5))
    return hasil


# ===================== 3. HISTOGRAM & OPERASI TITIK (02) =====================
def potong(v):
    # clipping ke 0..255 lalu dibulatkan ke bawah -> indeks histogram
    if v < 0:
        return 0
    if v > 255:
        return 255
    return int(v)


def histogram(data, mask=None):
    # h(r) = banyaknya piksel dengan graylevel r  (nilai di-clip ke 0..255)
    hist = [0] * 256
    if mask is None:
        nilai = data
    else:
        nilai = [v for v, m in zip(data, mask) if m]
    for v in nilai:
        if v < 0:
            hist[0] += 1
        elif v >= 255:
            hist[255] += 1
        else:
            hist[int(v)] += 1
    return hist


def statistik(hist):
    # rata-rata  mu = sum(i * h(i)) / N
    # std        sigma = sqrt( sum((i - mu)^2 * h(i)) / N )
    n = 0
    total = 0
    for i in range(256):
        n += hist[i]
        total += i * hist[i]
    if n == 0:
        return 0, 0
    rata = total / n
    kuadrat = 0
    for i in range(256):
        kuadrat += (i - rata) * (i - rata) * hist[i]
    return rata, (kuadrat / n) ** 0.5


def persentil(hist, p):
    # graylevel terkecil di mana p bagian piksel ada di bawahnya
    n = 0
    for v in hist:
        n += v
    kumulatif = 0
    for i in range(256):
        kumulatif += hist[i]
        if kumulatif >= p * n:
            return i
    return 255


def otsu(hist):
    # Threshold Otsu: pilih T yang memaksimalkan variansi antar-kelas
    #   sigma_B^2(T) = w0 * w1 * (mu0 - mu1)^2
    n = 0
    total = 0
    for i in range(256):
        n += hist[i]
        total += i * hist[i]
    terbaik = -1
    batas = 0
    n0 = 0
    total0 = 0
    for t in range(256):
        n0 += hist[t]
        total0 += t * hist[t]
        n1 = n - n0
        if n0 == 0 or n1 == 0:
            continue
        mu0 = total0 / n0
        mu1 = (total - total0) / n1
        varians = n0 * n1 * (mu0 - mu1) * (mu0 - mu1)
        if varians > terbaik:
            terbaik = varians
            batas = t
    return batas


def keterpisahan_otsu(hist, batas):
    # eta = variansi antar-kelas / variansi total   (0..1)
    # Mengukur seberapa baik sebuah citra fitur terbelah menjadi 2 kelompok
    # oleh threshold Otsu. Makin besar makin jelas pemisahannya.
    n = 0
    total = 0
    for i in range(256):
        n += hist[i]
        total += i * hist[i]
    rata = total / n
    var_total = 0
    for i in range(256):
        var_total += (i - rata) * (i - rata) * hist[i]
    n0 = 0
    total0 = 0
    for t in range(batas + 1):
        n0 += hist[t]
        total0 += t * hist[t]
    n1 = n - n0
    if n0 == 0 or n1 == 0 or var_total == 0:
        return 0.0
    mu0 = total0 / n0
    mu1 = (total - total0) / n1
    return n0 * n1 * (mu0 - mu1) * (mu0 - mu1) / n / var_total


def lut_stretching(bawah, atas):
    # Contrast stretching linear:  s = (r - bawah) * 255 / (atas - bawah)
    lut = []
    for r in range(256):
        if atas <= bawah:
            s = r
        else:
            s = (r - bawah) * 255 / (atas - bawah)
        lut.append(potong(s))
    return lut


def lut_gamma(gamma):
    # Gamma correction (power law):  s = 255 * (r / 255) ^ gamma
    return [255 * (r / 255) ** gamma for r in range(256)]


def lut_equalization(hist):
    # Histogram equalization:  s_k = 255 * sum_{j<=k} n_j / N   (CDF)
    n = 0
    for v in hist:
        n += v
    lut = []
    kumulatif = 0
    for k in range(256):
        kumulatif += hist[k]
        lut.append(255 * kumulatif / n)
    return lut


def pakai_lut(data, lut):
    # transformasi titik: setiap graylevel r diganti lut[r]
    return [lut[0] if v < 0 else (lut[255] if v >= 255 else lut[int(v)]) for v in data]


def normalisasi(data):
    # skala linear ke 0..255 (untuk ditampilkan / di-threshold)
    terkecil = data[0]
    terbesar = data[0]
    for v in data:
        if v < terkecil:
            terkecil = v
        if v > terbesar:
            terbesar = v
    rentang = terbesar - terkecil
    if rentang == 0:
        return [0.0] * len(data)
    return [(v - terkecil) * 255 / rentang for v in data]


# ========================= 4. SPATIAL FILTERING (03) =========================
def perlebar_tepi(data, lebar, tinggi, r):
    # "Perlebar citra, piksel di pinggir diisi perulangan nilai pinggir"
    lebar_baru = lebar + 2 * r
    hasil = []
    for yy in range(-r, tinggi + r):
        y = yy
        if y < 0:
            y = 0
        if y > tinggi - 1:
            y = tinggi - 1
        baris = data[y * lebar:(y + 1) * lebar]
        hasil.extend([baris[0]] * r + baris + [baris[-1]] * r)
    return hasil, lebar_baru


def jumlah_jendela(data, lebar, tinggi, k, hitung_luas=True):
    """Jumlah piksel di jendela k x k untuk SETIAP piksel (+ banyak piksel
    yang benar-benar ada di jendela, karena di pinggir jendela terpotong).

    Dihitung dengan jumlah kumulatif (prefix sum / citra integral) yang
    dipisah 2 arah (filter kotak itu separabel):
      1. arah horizontal : P(x) = f(0) + ... + f(x-1)
                           H(x) = P(x+r+1) - P(x-r)
      2. arah vertikal   : sama, tetapi pada hasil H antar-baris.
    Hasilnya sama persis dengan menjumlahkan k x k piksel satu per satu,
    tetapi tidak bergantung pada ukuran k (cepat untuk jendela besar).
    """
    r = k // 2
    xa = [x - r if x - r > 0 else 0 for x in range(lebar)]
    xb = [x + r + 1 if x + r + 1 < lebar else lebar for x in range(lebar)]
    # kumulatif vertikal dari hasil horizontal: C[y] = H[0] + ... + H[y-1]
    kumulatif = [[0.0] * lebar]
    for y in range(tinggi):
        baris = data[y * lebar:(y + 1) * lebar]
        jumlah = 0.0
        prefix = [0.0] + [jumlah := jumlah + v for v in baris]
        horizontal = [prefix[b] - prefix[a] for a, b in zip(xa, xb)]
        kumulatif.append([c + v for c, v in zip(kumulatif[-1], horizontal)])
    hasil = []
    luas = []
    lebar_jendela = [b - a for a, b in zip(xa, xb)]
    for y in range(tinggi):
        ya = y - r if y - r > 0 else 0
        yb = y + r + 1 if y + r + 1 < tinggi else tinggi
        hasil.extend([p - q for p, q in zip(kumulatif[yb], kumulatif[ya])])
        if hitung_luas:
            luas.extend([(yb - ya) * lw for lw in lebar_jendela])
    return hasil, luas


def filter_rata(data, lebar, tinggi, k):
    # LOWPASS / MEAN FILTER k x k (semua bobot 1/k^2, total bobot = 1):
    #   g(x,y) = (1/k^2) * sum f(x+i, y+j)
    # Di pinggir citra jendela dipotong (dibagi banyak piksel yang ada).
    jumlah, luas = jumlah_jendela(data, lebar, tinggi, k)
    return [s / n for s, n in zip(jumlah, luas)]


def std_lokal(data, lebar, tinggi, k):
    # Simpangan baku di jendela k x k:
    #   sigma = sqrt( mean(f^2) - mean(f)^2 )
    # kecil = daerah HALUS (permukaan obat), besar = KASAR (foil bertekstur)
    rata = filter_rata(data, lebar, tinggi, k)
    rata_kuadrat = filter_rata([v * v for v in data], lebar, tinggi, k)
    hasil = [0.0] * len(data)
    for i in range(len(data)):
        v = rata_kuadrat[i] - rata[i] * rata[i]
        hasil[i] = v ** 0.5 if v > 0 else 0.0
    return hasil


def konvolusi(data, lebar, tinggi, kernel):
    # g(x,y) = sum_i sum_j h(i,j) * f(x+j-r, y+i-r)   (sliding window)
    # Dikerjakan per baris: untuk setiap bobot h(i,j), baris citra yang
    # sudah digeser dikalikan bobot lalu dijumlahkan.
    k = len(kernel)
    r = k // 2
    p, lp = perlebar_tepi(data, lebar, tinggi, r)
    hasil = []
    for y in range(tinggi):
        acc = [0.0] * lebar
        for i in range(k):
            awal = (y + i) * lp
            for j in range(k):
                bobot = kernel[i][j]
                if bobot != 0:
                    geser = p[awal + j:awal + j + lebar]
                    acc = [a + bobot * v for a, v in zip(acc, geser)]
        hasil.extend(acc)
    return hasil


KERNEL_LOWPASS = [[1 / 16, 2 / 16, 1 / 16],
                  [2 / 16, 4 / 16, 2 / 16],
                  [1 / 16, 2 / 16, 1 / 16]]

def urutkan(daftar):
    # insertion sort (manual, tanpa sorted())
    a = list(daftar)
    for i in range(1, len(a)):
        kunci = a[i]
        j = i - 1
        while j >= 0 and a[j] > kunci:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = kunci
    return a


def filter_median(data, lebar, tinggi):
    # MEDIAN 3x3: 9 tetangga diurutkan, diambil nilai ke-5 (tengah)
    p, lp = perlebar_tepi(data, lebar, tinggi, 1)
    hasil = [0.0] * (lebar * tinggi)
    for y in range(tinggi):
        for x in range(lebar):
            a = (y + 1) * lp + x + 1
            jendela = [p[a - lp - 1], p[a - lp], p[a - lp + 1],
                       p[a - 1], p[a], p[a + 1],
                       p[a + lp - 1], p[a + lp], p[a + lp + 1]]
            hasil[y * lebar + x] = urutkan(jendela)[4]
    return hasil


# ================== 5. MORFOLOGI MASK BINER (09 REPRESENTATION) ==============
def jumlah_tetangga(mask, lebar, tinggi, k):
    # banyaknya piksel bernilai 1 di jendela k x k; di luar citra dianggap 0
    jumlah, _ = jumlah_jendela(mask, lebar, tinggi, k, hitung_luas=False)
    return jumlah


def dilasi(mask, lebar, tinggi, k):
    # DILASI (structuring element kotak k x k):
    #   piksel = 1 bila MINIMAL SATU piksel di jendela bernilai 1
    #   (= filter MAKSIMUM).  Objek menebal.
    n = jumlah_tetangga(mask, lebar, tinggi, k)
    return [1 if v > 0.5 else 0 for v in n]


def erosi(mask, lebar, tinggi, k):
    # EROSI: piksel tetap 1 hanya bila SEMUA piksel di jendela bernilai 1
    #   (= filter MINIMUM).  Objek menipis.  Di luar citra dianggap 1
    #   supaya objek yang menyentuh tepi tidak ikut terkikis.
    r = k // 2
    terbalik = [1 - v for v in mask]
    n = jumlah_tetangga(terbalik, lebar, tinggi, k)
    return [0 if v > 0.5 else 1 for v in n]


def opening(mask, lebar, tinggi, k):
    # OPENING = erosi lalu dilasi -> buang bintik kecil & putus jembatan tipis
    return dilasi(erosi(mask, lebar, tinggi, k), lebar, tinggi, k)


def closing(mask, lebar, tinggi, k):
    # CLOSING = dilasi lalu erosi -> tutup celah & lubang kecil
    return erosi(dilasi(mask, lebar, tinggi, k), lebar, tinggi, k)


def kontur(mask, lebar, tinggi):
    # garis batas objek = mask - erosi(mask)
    e = erosi(mask, lebar, tinggi, 3)
    return [mask[i] - e[i] for i in range(len(mask))]


# ============== 6. CONNECTED COMPONENT / REGION (08 SEGMENTATION) ============
TETANGGA_8 = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))


def label_komponen(mask, lebar, tinggi):
    # Pelabelan komponen terhubung (8-tetangga) dengan region growing:
    # mulai dari satu piksel objek (seed), lalu terus menambahkan
    # tetangga yang juga objek, memakai tumpukan (stack).
    label = [0] * (lebar * tinggi)
    daftar = []
    nomor = 0
    for awal in range(lebar * tinggi):
        if mask[awal] == 0 or label[awal] != 0:
            continue
        nomor += 1
        label[awal] = nomor
        tumpukan = [awal]
        piksel = []
        while tumpukan:
            i = tumpukan.pop()
            piksel.append(i)
            x = i % lebar
            y = i // lebar
            for (dx, dy) in TETANGGA_8:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < lebar and 0 <= ny < tinggi:
                    j = ny * lebar + nx
                    if mask[j] and label[j] == 0:
                        label[j] = nomor
                        tumpukan.append(j)
        daftar.append({"nomor": nomor, "piksel": piksel, "luas": len(piksel)})
    return label, daftar


def isi_lubang(mask, lebar, tinggi):
    # Lubang = daerah 0 yang TIDAK terhubung ke tepi citra -> diisi 1.
    # Dicari dengan flood fill dari semua piksel 0 di tepi citra.
    luar = [0] * (lebar * tinggi)
    tumpukan = []
    for x in range(lebar):
        for y in (0, tinggi - 1):
            i = y * lebar + x
            if mask[i] == 0 and luar[i] == 0:
                luar[i] = 1
                tumpukan.append(i)
    for y in range(tinggi):
        for x in (0, lebar - 1):
            i = y * lebar + x
            if mask[i] == 0 and luar[i] == 0:
                luar[i] = 1
                tumpukan.append(i)
    while tumpukan:
        i = tumpukan.pop()
        x = i % lebar
        y = i // lebar
        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < lebar and 0 <= ny < tinggi:
                j = ny * lebar + nx
                if mask[j] == 0 and luar[j] == 0:
                    luar[j] = 1
                    tumpukan.append(j)
    return [1 - v for v in luar]


def region_ke_mask(piksel, n):
    m = [0] * n
    for i in piksel:
        m[i] = 1
    return m


def komponen_terbesar(mask, lebar, tinggi):
    _, daftar = label_komponen(mask, lebar, tinggi)
    hasil = [0] * (lebar * tinggi)
    if not daftar:
        return hasil
    besar = daftar[0]
    for d in daftar:
        if d["luas"] > besar["luas"]:
            besar = d
    for i in besar["piksel"]:
        hasil[i] = 1
    return hasil


def bungkus_oktagon(mask, lebar, tinggi):
    # Pembungkus oktagon: batas min/maks dari x, y, x+y, x-y.
    # (pendekatan convex hull dengan 8 sisi). Dipakai karena blister
    # berbentuk cembung, sehingga coakan akibat silau/obat di pinggir
    # ikut tertutup.
    nilai_min = [None, None, None, None]
    nilai_maks = [None, None, None, None]
    for i in range(lebar * tinggi):
        if mask[i]:
            x = i % lebar
            y = i // lebar
            ciri = (x, y, x + y, x - y)
            for k in range(4):
                if nilai_min[k] is None or ciri[k] < nilai_min[k]:
                    nilai_min[k] = ciri[k]
                if nilai_maks[k] is None or ciri[k] > nilai_maks[k]:
                    nilai_maks[k] = ciri[k]
    hasil = [0] * (lebar * tinggi)
    if nilai_min[0] is None:
        return hasil
    for y in range(tinggi):
        for x in range(lebar):
            ciri = (x, y, x + y, x - y)
            dalam = 1
            for k in range(4):
                if ciri[k] < nilai_min[k] or ciri[k] > nilai_maks[k]:
                    dalam = 0
                    break
            hasil[y * lebar + x] = dalam
    return hasil


def kotak_pembatas(mask, lebar, tinggi):
    # bounding box (x0, y0, x1, y1) dari piksel bernilai 1
    x0 = lebar
    y0 = tinggi
    x1 = -1
    y1 = -1
    for i in range(lebar * tinggi):
        if mask[i]:
            x = i % lebar
            y = i // lebar
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
    if x1 < 0:
        return 0, 0, lebar, tinggi
    return x0, y0, x1 + 1, y1 + 1


# ===================== 7. REPRESENTASI & DESKRIPTOR (09) =====================
ARAH_RANTAI = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))


def kode_rantai(piksel, lebar):
    """Kode rantai Freeman 8-arah dari batas luar sebuah region.

    Arah: 0=kanan, 1=kanan-atas, 2=atas, 3=kiri-atas, 4=kiri,
          5=kiri-bawah, 6=bawah, 7=kanan-bawah.
    Penelusuran batas (boundary following):
      * mulai dari piksel paling atas, lalu paling kiri
      * arah terakhir d; pencarian tetangga berikutnya dimulai dari
        (d + 7) mod 8 bila d genap, (d + 6) mod 8 bila d ganjil,
        berputar berlawanan jarum jam sampai menemukan piksel objek
      * selesai ketika kembali ke piksel awal
    """
    ada = set(piksel)
    awal = piksel[0]
    for i in piksel:
        if i < awal:
            awal = i
    x0 = awal % lebar
    y0 = awal // lebar
    x = x0
    y = y0
    arah = 7
    kode = []
    for _ in range(4 * len(piksel) + 8):
        mulai = (arah + 7) % 8 if arah % 2 == 0 else (arah + 6) % 8
        ketemu = False
        for k in range(8):
            d = (mulai + k) % 8
            nx = x + ARAH_RANTAI[d][0]
            ny = y + ARAH_RANTAI[d][1]
            if 0 <= nx < lebar and ny >= 0 and (ny * lebar + nx) in ada:
                kode.append(d)
                x = nx
                y = ny
                arah = d
                ketemu = True
                break
        if not ketemu or (x == x0 and y == y0):
            break
    return kode


def keliling_dari_rantai(kode):
    # arah genap (horizontal/vertikal) panjang 1, arah ganjil (diagonal) sqrt(2)
    p = 0.0
    for a in kode:
        p += 1.0 if a % 2 == 0 else 1.41421356
    return p


def momen(piksel, lebar):
    """Momen region.
    pusat massa : cx = m10/m00, cy = m01/m00
    momen pusat : mu20, mu02, mu11 (dibagi luas)
    sumbu elips : lambda1,2 = (mu20+mu02)/2 +- sqrt(((mu20-mu02)/2)^2 + mu11^2)
    lonjong     = sqrt(lambda1 / lambda2)        (1 = bulat)
    isi_elips   = luas / (4*pi*sqrt(lambda1*lambda2))   (~1 untuk bentuk
                  elips/kapsul yang utuh, kecil untuk bentuk aneh)
    """
    n = len(piksel)
    sx = 0
    sy = 0
    for i in piksel:
        sx += i % lebar
        sy += i // lebar
    cx = sx / n
    cy = sy / n
    m20 = 0.0
    m02 = 0.0
    m11 = 0.0
    for i in piksel:
        dx = i % lebar - cx
        dy = i // lebar - cy
        m20 += dx * dx
        m02 += dy * dy
        m11 += dx * dy
    m20 /= n
    m02 /= n
    m11 /= n
    tengah = (m20 + m02) / 2
    akar = (((m20 - m02) / 2) ** 2 + m11 * m11) ** 0.5
    l1 = tengah + akar + 1 / 12        # +1/12: koreksi ukuran piksel
    l2 = tengah - akar + 1 / 12
    if l2 < 0.01:
        l2 = 0.01
    pi = 3.14159265
    isi = n / (4 * pi * (l1 * l2) ** 0.5)
    lonjong = (l1 / l2) ** 0.5
    return cx, cy, isi, lonjong


# ============ 8. ESTIMASI LATAR LOKAL (MEDIAN PER BLOK, ORDER STATISTIC) ======
def median_per_blok(data, mask, lebar, tinggi, ukuran):
    """Median lokal dari piksel mask, dihitung per blok ukuran x ukuran.

    1. Citra dibagi menjadi blok-blok. Untuk setiap blok dibuat histogram
       piksel yang mask-nya 1, lalu diambil median (persentil 50%).
       Median (order statistic) tidak terpengaruh piksel ekstrem seperti
       kilau di tepi kantong, berbeda dengan rata-rata.
    2. Blok yang terlalu sedikit pikselnya diisi dari blok tetangga
       (diulang sampai semua terisi).
    3. Hasil per blok dikembalikan ke ukuran citra lalu dihaluskan dengan
       mean filter supaya tidak ada tepi blok.
    """
    nbx = (lebar + ukuran - 1) // ukuran
    nby = (tinggi + ukuran - 1) // ukuran
    hist = [[0] * 256 for _ in range(nbx * nby)]
    for y in range(tinggi):
        by = (y // ukuran) * nbx
        awal = y * lebar
        for x in range(lebar):
            if mask[awal + x]:
                v = data[awal + x]
                hist[by + x // ukuran][0 if v < 0 else (255 if v >= 255 else int(v))] += 1
    nilai = [None] * (nbx * nby)
    minimal = ukuran * ukuran // 10
    for b in range(nbx * nby):
        n = 0
        for c in hist[b]:
            n += c
        if n >= minimal:
            nilai[b] = persentil(hist[b], 0.5)
    if all(v is None for v in nilai):
        return [128.0] * (lebar * tinggi)
    while any(v is None for v in nilai):
        baru = list(nilai)
        for by in range(nby):
            for bx in range(nbx):
                if nilai[by * nbx + bx] is not None:
                    continue
                total = 0
                cacah = 0
                for (dx, dy) in TETANGGA_8:
                    tx = bx + dx
                    ty = by + dy
                    if 0 <= tx < nbx and 0 <= ty < nby and nilai[ty * nbx + tx] is not None:
                        total += nilai[ty * nbx + tx]
                        cacah += 1
                if cacah > 0:
                    baru[by * nbx + bx] = total / cacah
        nilai = baru
    peta = [0.0] * (lebar * tinggi)
    for y in range(tinggi):
        by = (y // ukuran) * nbx
        for x in range(lebar):
            peta[y * lebar + x] = nilai[by + x // ukuran]
    return filter_rata(peta, lebar, tinggi, ukuran + 1)
