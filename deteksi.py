# =============================================================================
# deteksi.py  -  MENGHITUNG SISA OBAT DI DALAM BLISTER DARI SEBUAH FOTO
#
# PROGRAM UTAMA PENGOLAHAN CITRA. File ini dan folder dip/ TIDAK meng-import
# library apa pun (tidak ada numpy, opencv, math, dll). Yang di-import hanya
# modul buatan sendiri di folder dip/ (satu file per materi kuliah).
# Tidak ada Machine Learning: semua keputusan memakai rumus dan aturan yang
# bisa dijelaskan (threshold, morfologi, deskriptor region).
#
# Masukan : lebar, tinggi, data (bytes R,G,B,R,G,B,... dari masukan.py)
# Keluaran: dict hasil -> jumlah obat, posisi & bentuk setiap obat, region
#           yang ditolak beserta alasannya, dan DAFTAR LANGKAH (gambar +
#           rumus + penjelasan) untuk ditampilkan satu per satu di GUI.
#
# ALUR BESAR
#   TAHAP A (foto utuh diperkecil)       -> di mana letak blister?
#     A1 sampling  A2 YIQ & kroma  A3 warna latar + Otsu  A4 ROI blister
#     A5 potong blister dari foto asli
#   TAHAP B (potongan blister)           -> berapa obat yang masih ada?
#     B1 restorasi  B2 grayscale & histogram  B3 tekstur (std lokal)
#     B4 top-hat (lebih terang dari foil)  B5 beda warna dari foil
#     B6 kandidat obat  B7 labeling + penilaian + region splitting
#     B8 gabung belahan kapsul  B9 konsistensi ukuran
#     B10 chain code & deskripsi bentuk  B11 hasil akhir
# =============================================================================
from dip.dasar import sampling, sampling_kotak, faktor_sampling, akar, nilai_tengah, ke_int
from dip.m02_enhancement import histogram, rata_dan_std, skala_8bit
from dip.m03_spasial import std_lokal, konvolusi3, median3, LOWPASS_32
from dip.m05_warna import ke_abu, rgb_ke_hsi, median_warna, jarak_warna
from dip.m06_restorasi import cek_salt_pepper, estimasi_noise
from dip.m08_segmentasi import otsu, label_region, mask_dari_region
from dip.m09_representasi import (opening, closing, dilasi, erosi, isi_lubang, momen, top_hat,
                                  chain_code, keliling)
from dip.dasar import PI, arctan2


# ================================ PARAMETER ==================================
UKURAN_A = 400          # sisi terpanjang foto utuh pada tahap A (cari blister)
UKURAN_B = 640          # skala kerja tahap B: setara foto utuh 640 piksel
PITA_TEPI = 25          # pita tepi foto = 1/25 sisi terpendek (dianggap latar)
SKALA_KROMA = 60.0      # jarak kroma 0..60 dipetakan ke 0..255 untuk histogram
SKALA_STD = 40.0        # std lokal 0..40 dipetakan ke 0..255
SKALA_TOPHAT = 100.0    # top-hat 0..100 dipetakan ke 0..255
JENDELA_STD = 5         # jendela simpangan baku lokal 5 x 5

# restorasi (materi 06)
BATAS_SALT = 0.01       # > 1% piksel impulse -> median filter
BATAS_NOISE = 6.0       # noise patch datar > 6 -> lowpass 1/32

# kandidat (materi 02, 05, 09)
BATAS_FOIL = 0.85       # di top-hat, piksel lebih gelap dari 0.85 x foil dianggap = foil
MIN_T_TOPHAT = 12.0     # threshold top-hat minimal (gray level)
MIN_T_KROMA = 8.0       # threshold beda kroma minimal
GELAP_BOLEH = 0.6       # beda warna hanya dihitung bila Y > 0.6 x foil ...
JENUH_GELAP = 0.35      # ... atau sangat jenuh (S > 0.35) dan Y > 0.25 x foil

# penilaian region (materi 09)
MIN_LUAS = 100          # region < 100 piksel (skala kerja) bukan obat
MIN_LUAS_ROI = 0.0015   # ... atau < 0.15% luas blister
MAKS_LUAS_ROI = 0.12    # satu obat tidak mungkin > 12% luas blister
MIN_ISI = 0.82          # keterisian elips minimal (bentuk rapi)
MAKS_LONJONG = 5.0      # kelonjongan maksimal
LEBAR_CINCIN = 6        # lebar cincin pembanding di sekeliling region (px)
MIN_DI_ROI = 0.6        # >= 60% cincin harus di dalam blister
MAKS_CINCIN_FOIL = 20.0 # warna cincin harus mirip foil (beda kroma <= 20)
BEDA_TERANG = 15        # obat putih: lebih terang >= 15 dari cincinnya ...
TERANG_FOIL = 1.14      # ... dan >= 1.14 x kecerahan foil
BEDA_KROMA = 10         # obat berwarna: beda kroma >= 10 dari cincinnya ...
JENUH = 0.25            # ... boleh lebih gelap bila jenuh (S >= 0.25)
KROMA_KUAT = 25         # ... atau warnanya sangat beda (>= 25)
MIRIP_LATAR = 6.0       # kroma mirip latar (< 6) & tidak terang -> kantong kosong
MAKS_TINGKAT = 3        # region splitting paling dalam 3 tingkat

# konsistensi ukuran (materi 09)
KECIL_ACUAN = 0.4       # luas < 0.4 x acuan -> serpihan
BESAR_ACUAN = 2.2       # luas > 2.2 x acuan -> bukan satu obat

# kapsul
WARNA_KAPSUL = 12       # belahan berwarna: beda kroma >= 12 atau S >= 0.25
LONJONG_BULAT = 1.6     # kelonjongan < 1.6 -> bulat (kaplet/kapsul >= 2.4)


# ============================== ALAT BANTU ===================================
class Konteks:
    # wadah data potongan blister yang dipakai banyak fungsi
    pass


def ganjil(x):
    # ukuran structuring element harus ganjil (punya titik tengah)
    x = int(x + 0.5)
    if x < 1:
        x = 1
    return x if x % 2 == 1 else x + 1


def bulat(x, digit=1):
    # angka -> teks pendek untuk penjelasan
    if digit == 0:
        return str(int(x + 0.5)) if x >= 0 else str(-int(-x + 0.5))
    return str(round(x, digit))


def kotak_lokal(piksel, r, w, h):
    # kotak kecil di sekitar sekumpulan piksel (dengan margin r) + mask di
    # dalam kotak. Morfologi dikerjakan di kotak ini saja supaya cepat.
    xs = [i % w for i in piksel]
    ys = [i // w for i in piksel]
    x0 = max(min(xs) - r, 0)
    y0 = max(min(ys) - r, 0)
    x1 = min(max(xs) + r + 1, w)
    y1 = min(max(ys) + r + 1, h)
    bw = x1 - x0
    bh = y1 - y0
    m = [0] * (bw * bh)
    for x, y in zip(xs, ys):
        m[(y - y0) * bw + (x - x0)] = 1
    return x0, y0, bw, bh, m


def ke_global(daftar, x0, y0, bw, w):
    # indeks kotak lokal -> indeks gambar
    return [(y0 + q // bw) * w + x0 + q % bw for q in daftar]


def otsu_eta(nilai8):
    # Otsu pada sekumpulan nilai 0..255. Selain T, dihitung juga
    # eta = sigma_B^2 / sigma_total^2 (0..1): seberapa jelas dua kelas terpisah.
    hist = [0] * 256
    for v in nilai8:
        hist[v] += 1
    T = otsu(hist)
    n = len(nilai8)
    mu = sum(nilai8) / float(n)
    var = sum((v - mu) ** 2 for v in nilai8) / n
    a = [v for v in nilai8 if v <= T]
    b = [v for v in nilai8 if v > T]
    if var <= 0 or not a or not b:
        return T, 0.0
    ma = sum(a) / float(len(a))
    mb = sum(b) / float(len(b))
    sb = (len(a) / float(n)) * (len(b) / float(n)) * (ma - mb) ** 2
    return T, sb / var


def kroma_yiq(R, G, B):
    # YIQ (materi 05): Y = terang-gelap, I dan Q = warna.
    # Kroma TERNORMALISASI c1 = 100 I / (Y + 20), c2 = 100 Q / (Y + 20):
    # warna yang sama dengan terang berbeda (bayangan / lampu) tetap mirip.
    # (+20 supaya piksel yang sangat gelap tidak meledak nilainya.)
    Y = ke_abu(R, G, B)
    c1 = [(0.596 * r - 0.275 * g - 0.321 * b) * 100.0 / (y + 20.0) for r, g, b, y in zip(R, G, B, Y)]
    c2 = [(0.212 * r - 0.523 * g + 0.311 * b) * 100.0 / (y + 20.0) for r, g, b, y in zip(R, G, B, Y)]
    return Y, c1, c2


def langkah_baru(judul, materi, teknik, rumus, deskripsi, gambar, dipakai=True):
    return {"judul": judul, "materi": materi, "teknik": teknik, "rumus": rumus,
            "deskripsi": deskripsi, "gambar": gambar, "dipakai": dipakai}


# ---- gambar untuk ditampilkan GUI (data mentah, GUI yang menggambar) ----
def g_rgb(w, h, R, G, B):
    return {"jenis": "rgb", "w": w, "h": h, "R": R, "G": G, "B": B}


def g_abu(w, h, data8):
    return {"jenis": "gray", "w": w, "h": h, "data": data8}


def g_mask(w, h, mask):
    return {"jenis": "mask", "w": w, "h": h, "data": [255 if v else 0 for v in mask]}


def g_panas(w, h, data8, keterangan):
    return {"jenis": "heat", "w": w, "h": h, "data": data8, "skala": keterangan}


def g_hist(hist, garis, label, nama_garis="T"):
    return {"jenis": "hist", "hist": hist, "garis": garis, "label": label, "nama_garis": nama_garis}


def g_overlay(c, kelompok, redup=0.55):
    # foto potongan diredupkan, lalu batas region (A - erosi(A), materi 09)
    # diwarnai. kelompok = [(daftar_region, (r, g, b), isi_tipis), ...]
    R = [int(v * redup) for v in c.R]
    G = [int(v * redup) for v in c.G]
    B = [int(v * redup) for v in c.B]
    for daftar, warna, isi in kelompok:
        for d in daftar:
            x0, y0, bw, bh, m = kotak_lokal(d["piksel"], 1, c.w, c.h)
            e = erosi(m, bw, bh, 3)
            for q in range(bw * bh):
                if not m[q]:
                    continue
                i = (y0 + q // bw) * c.w + x0 + q % bw
                if not e[q]:
                    R[i], G[i], B[i] = warna
                elif isi:
                    R[i] = (R[i] + warna[0]) // 2
                    G[i] = (G[i] + warna[1]) // 2
                    B[i] = (B[i] + warna[2]) // 2
    return g_rgb(c.w, c.h, R, G, B)


# ======================= TAHAP A : MENCARI BLISTER ===========================
def mask_roi(w, h, c1, c2, R, G, B, acuan, skala):
    # Piksel dianggap BUKAN LATAR bila warnanya cukup jauh dari warna latar:
    #   jarak kroma  sqrt((c1-b1)^2 + (c2-b2)^2) > Tk     (beda warna)
    #   ATAU jarak RGB Euclid ke warna latar > Tr          (beda terang)
    # Lalu dirapikan dengan morfologi (materi 09):
    #   opening -> buang bintik kecil, closing -> satukan foil yang terputus,
    #   hole filling -> obat & kantong di dalam blister ikut menjadi ROI.
    b1, b2, latar_rgb, tk, tr = acuan
    jr = jarak_warna(R, G, B, latar_rgb)
    roi = [1 if (((a - b1) ** 2 + (b - b2) ** 2) ** 0.5 > tk or d > tr) else 0
           for a, b, d in zip(c1, c2, jr)]
    roi = opening(roi, w, h, ganjil(5 * skala))
    roi = closing(roi, w, h, ganjil(9 * skala))
    roi = isi_lubang(roi, w, h)
    return roi


def tahap_a(lebar0, tinggi0, data, langkah):
    # ---------------------------------------------------------------- A1
    wa, ha, Ra, Ga, Ba, fa = sampling(lebar0, tinggi0, data, UKURAN_A)
    na = wa * ha
    langkah.append(langkah_baru(
        "Foto asli & sampling", "01 Introduction to DIP I",
        "Citra digital f(x,y) + sampling (digitisasi koordinat)",
        "g(m,n) = [f(a,a) + f(a,b) + f(b,a) + f(b,b)] / 4,  blok " + str(fa) + " x " + str(fa),
        "Foto dibaca sebagai matriks " + str(lebar0) + " x " + str(tinggi0) + " piksel, 3 kanal R, G, B "
        "(8 bit, 0..255). Untuk MENCARI letak blister foto cukup kecil, jadi setiap blok " + str(fa) + " x " +
        str(fa) + " piksel diwakili rata-rata 4 titik sampel -> " + str(wa) + " x " + str(ha) + " piksel. "
        "Rata-rata 4 titik = lowpass kecil sebelum sampling sehingga noise dan pola halus foil tidak "
        "menimbulkan aliasing.",
        g_rgb(wa, ha, Ra, Ga, Ba)))

    # ---------------------------------------------------------------- A2
    Ya, c1a, c2a = kroma_yiq(Ra, Ga, Ba)
    p = max(3, min(wa, ha) // PITA_TEPI)
    pita = [1 if (i % wa) < p or (i % wa) >= wa - p or (i // wa) < p or (i // wa) >= ha - p else 0
            for i in range(na)]
    tepi = [i for i in range(na) if pita[i]]
    b1 = nilai_tengah([c1a[i] for i in tepi])
    b2 = nilai_tengah([c2a[i] for i in tepi])
    latar_rgb = median_warna(Ra, Ga, Ba, tepi)
    jk = [((a - b1) ** 2 + (b - b2) ** 2) ** 0.5 for a, b in zip(c1a, c2a)]
    jk8 = skala_8bit(jk, 0, SKALA_KROMA)
    langkah.append(langkah_baru(
        "Ruang warna YIQ & kroma ternormalisasi", "05 Color Image Processing",
        "RGB -> YIQ, kroma c1 = I/(Y+20), c2 = Q/(Y+20)",
        "Y = 0.299R + 0.587G + 0.114B;  I = 0.596R - 0.275G - 0.321B;  Q = 0.212R - 0.523G + 0.311B",
        "Pada YIQ, Y menyatakan terang-gelap sedangkan I dan Q menyatakan warna (slide 05). I dan Q "
        "dibagi (Y + 20) supaya warna yang sama tetap mirip walaupun terkena bayangan atau lampu. "
        "Latar (meja kayu, kain, kulit tangan) berwarna HANGAT, sedangkan foil blister dan obat "
        "abu-abu/kebiruan. Warna latar diambil dari pita " + str(p) + " px di tepi foto (median): "
        "RGB " + str(latar_rgb) + ", kroma (" + bulat(b1) + ", " + bulat(b2) + "). Gambar: jarak kroma "
        "setiap piksel ke warna latar (terang = beda dari latar).",
        g_panas(wa, ha, jk8, "jarak kroma ke latar (0..60)")))

    # ---------------------------------------------------------------- A3
    hist_k = histogram(jk8)
    t_k = otsu(hist_k)
    hist_r = histogram(skala_8bit(jarak_warna(Ra, Ga, Ba, latar_rgb), 0, 255))
    t_r = otsu(hist_r)
    acuan = (b1, b2, latar_rgb, t_k * SKALA_KROMA / 255.0, float(t_r))
    langkah.append(langkah_baru(
        "Threshold latar dengan Otsu", "02 Image Enhancement + 08 Image Segmentation",
        "Histogram + thresholding, T dipilih otomatis dengan Otsu (Gonzalez)",
        "p(r_k) = n_k / n;   T = argmax  w1(T) w2(T) [mu1(T) - mu2(T)]^2",
        "Histogram jarak kroma ke latar mempunyai dua kelompok: latar (dekat 0) dan blister (jauh). "
        "Threshold T dipilih dengan metode Otsu (buku Gonzalez, di luar slide): T yang membuat variansi "
        "ANTAR-kelas terbesar. Otsu murni rumus statistik histogram, tidak ada data latih (bukan machine "
        "learning). Hasil: T kroma = " + bulat(acuan[3]) + ". Dengan cara yang sama untuk jarak RGB "
        "(Euclid, materi 02) didapat T = " + str(t_r) + ".",
        g_hist(hist_k, t_k, "jarak kroma ke latar (0..255)")))

    # ---------------------------------------------------------------- A4
    roi_a = mask_roi(wa, ha, c1a, c2a, Ra, Ga, Ba, acuan, 1.0)
    _, daftar = label_region(roi_a, wa, ha, 4)
    pilih = []
    for d in daftar:
        # region kecil (< 0.5% foto) atau yang sebagian besar ada di pita tepi
        # (latar yang warnanya lain, mis. pinggir meja) bukan blister
        di_pita = sum(pita[i] for i in d["piksel"])
        if d["luas"] >= 0.005 * na and di_pita <= 0.3 * d["luas"]:
            pilih.append(d)
    roi_bersih = mask_dari_region(pilih, wa, ha)
    if pilih:
        x0 = min(d["x0"] for d in pilih)
        y0 = min(d["y0"] for d in pilih)
        x1 = max(d["x1"] for d in pilih) + 1
        y1 = max(d["y1"] for d in pilih) + 1
    else:
        x0, y0, x1, y1 = 0, 0, wa, ha
    langkah.append(langkah_baru(
        "Area blister (ROI)", "09 Representation (morfologi) + 02 Konektivitas",
        "Threshold ganda (kroma ATAU RGB), opening, closing, hole filling, pelabelan N4",
        "ROI = (jarak kroma > " + bulat(acuan[3]) + ") OR (jarak RGB > " + str(t_r) + ");  opening 5x5;  "
        "closing 9x9;  isi lubang",
        "Piksel yang tidak mirip latar dijadikan 1. Opening 5x5 (erosi lalu dilasi) membuang bintik, "
        "closing 9x9 (dilasi lalu erosi) menyatukan bagian foil yang terputus, lalu lubang di dalamnya "
        "(obat, kantong) diisi (hole filling, Gonzalez). Pelabelan N4 menemukan " + str(len(daftar)) +
        " region; yang dipakai " + str(len(pilih)) + " region besar yang tidak menempel di tepi foto. "
        "Tangan / jari berwarna hangat seperti latar sehingga ikut terbuang.",
        g_mask(wa, ha, roi_bersih)))

    # ---------------------------------------------------------------- A5
    m = max(2, (x1 - x0 + y1 - y0) // 30)
    kx0 = max((x0 - m) * fa, 0)
    ky0 = max((y0 - m) * fa, 0)
    kx1 = min((x1 + m) * fa, lebar0)
    ky1 = min((y1 + m) * fa, tinggi0)
    return acuan, fa, (kx0, ky0, kx1, ky1)


# ======================= TAHAP B : MENGHITUNG OBAT ===========================
def tahap_b_awal(lebar0, tinggi0, data, acuan, fa, kotak, langkah):
    c = Konteks()
    kx0, ky0, kx1, ky1 = kotak
    fb = faktor_sampling(lebar0, tinggi0, UKURAN_B)
    w, h, R, G, B, f = sampling_kotak(lebar0, tinggi0, data, kx0, ky0, kx1, ky1, UKURAN_B, fb)
    c.w, c.h, c.f = w, h, f
    c.kotak = kotak
    langkah.append(langkah_baru(
        "Potong blister dari foto asli", "01 Introduction to DIP I",
        "Cropping + sampling ulang",
        "kotak ROI x " + str(fa) + " -> foto asli [" + str(kx0) + ".." + str(kx1) + "] x [" + str(ky0) +
        ".." + str(ky1) + "], blok " + str(f) + " x " + str(f),
        "Kotak pembatas ROI dikali faktor sampling tahap A (" + str(fa) + ") supaya menjadi koordinat foto "
        "ASLI, ditambah sedikit margin, lalu hanya bagian itu yang diambil dan di-sampling dengan blok " +
        str(f) + " x " + str(f) + " -> " + str(w) + " x " + str(h) + " piksel. Semua langkah berikutnya "
        "bekerja pada potongan ini (lebih detail daripada foto kecil, dan lebih cepat daripada foto utuh).",
        g_rgb(w, h, R, G, B)))

    # ---------------------------------------------------------------- B1
    Y = ke_abu(R, G, B)
    Y8 = ke_int(Y)
    salt, med = cek_salt_pepper(Y8, w, h)
    noise = estimasi_noise(Y, w, h)
    pakai_median = salt > BATAS_SALT
    pakai_lowpass = noise > BATAS_NOISE
    if pakai_median:
        R = median3(R, w, h)
        G = median3(G, w, h)
        B = median3(B, w, h)
    if pakai_lowpass:
        R = ke_int(konvolusi3(R, w, h, LOWPASS_32, 32))
        G = ke_int(konvolusi3(G, w, h, LOWPASS_32, 32))
        B = ke_int(konvolusi3(B, w, h, LOWPASS_32, 32))
    teks = []
    if pakai_median:
        teks.append("median 3x3 dijalankan pada R, G, B")
    if pakai_lowpass:
        teks.append("lowpass (1/32)[1 3 1; 3 16 3; 1 3 1] dijalankan pada R, G, B")
    langkah.append(langkah_baru(
        "Cek noise -> restorasi", "06 Image Restoration (+ 03 Spatial Filtering)",
        "Deteksi salt & pepper + estimasi noise di patch datar",
        "g = f + eta;  impulse = |f - median3x3| > 60 di piksel ekstrem;  "
        "noise = rata |f - rata N4| di 25% blok 8x8 terdatar",
        "Sebelum dianalisis, citra diperiksa: piksel salt & pepper = " + bulat(salt * 100, 3) +
        "% (batas " + bulat(BATAS_SALT * 100, 0) + "%), noise di patch datar = " + bulat(noise, 2) +
        " (batas " + bulat(BATAS_NOISE, 0) + "). " +
        ("Dipakai: " + "; ".join(teks) + "." if teks else
         "Keduanya di bawah batas, jadi filter TIDAK dijalankan agar tepi obat tetap tajam."),
        g_abu(w, h, Y8), pakai_median or pakai_lowpass))

    # ---------------------------------------------------------------- B2
    Y, c1, c2 = kroma_yiq(R, G, B)
    Y8 = ke_int(Y)
    _, S, _ = rgb_ke_hsi(R, G, B)
    c.R, c.G, c.B, c.Y, c.c1, c.c2, c.S = R, G, B, Y, c1, c2, S
    c.b1, c.b2 = acuan[0], acuan[1]
    roi = mask_roi(w, h, c1, c2, R, G, B, acuan, fa / float(f))
    _, daftar = label_region(roi, w, h, 4)
    roi = mask_dari_region([d for d in daftar if d["luas"] >= 0.02 * w * h], w, h)
    luas_roi = sum(roi)
    if luas_roi == 0:
        roi = [1] * (w * h)
        luas_roi = w * h
    c.roi = roi
    c.luas_roi = luas_roi
    c.min_luas = max(MIN_LUAS, int(MIN_LUAS_ROI * luas_roi))
    hist_y = histogram(Y8, roi)
    mu, sd = rata_dan_std(hist_y)
    langkah.append(langkah_baru(
        "Grayscale (Y) & histogram blister", "05 Color Image Processing + 02 Image Enhancement",
        "Graylevel = luminance Y dari YIQ; histogram di dalam ROI",
        "Y = 0.299 R + 0.587 G + 0.114 B;   p(r_k) = n_k / n;   mu = sum k p(k)",
        "Potongan diubah ke graylevel dengan rumus luminance Y (slide 05: Y = terang-gelap). ROI "
        "dihitung ulang di potongan dengan threshold yang sama dari tahap A. Histogram graylevel di dalam "
        "blister: rata-rata " + bulat(mu, 0) + ", simpangan baku " + bulat(sd, 1) + ". Kecerahan setiap "
        "foto berbeda (terang/gelap), karena itu langkah berikutnya tidak memakai threshold tetap, "
        "melainkan membandingkan obat dengan foil DI FOTO YANG SAMA.",
        g_hist(hist_y, int(mu + 0.5), "graylevel Y di dalam blister", "rata-rata")))
    return c


def tahap_b_peta(c, langkah):
    w, h, n = c.w, c.h, c.w * c.h
    roi = c.roi
    # ---------------------------------------------------------------- B3
    sd = std_lokal(c.Y, w, h, JENDELA_STD)
    sd8 = skala_8bit(sd, 0, SKALA_STD)
    t_sd = otsu(histogram(sd8, roi))
    c.sd = sd
    kasar = [i for i in range(n) if roi[i] and sd8[i] > t_sd]
    Yf = nilai_tengah([c.Y[i] for i in kasar]) if kasar else 128.0
    c1f = nilai_tengah([c.c1[i] for i in kasar]) if kasar else 0.0
    c2f = nilai_tengah([c.c2[i] for i in kasar]) if kasar else 0.0
    c.Yf, c.c1f, c.c2f = Yf, c1f, c2f
    langkah.append(langkah_baru(
        "Tekstur: simpangan baku lokal", "03 Spatial Filtering + 08 Image Segmentation",
        "Dua lowpass rata-rata 5x5 -> std lokal, threshold Otsu",
        "std = sqrt( mean5x5(f^2) - mean5x5(f)^2 );  halus bila std <= T = " + bulat(t_sd * SKALA_STD / 255.0),
        "Foil aluminium bertekstur (pola belah ketupat, tepi kantong), permukaan obat HALUS. Simpangan "
        "baku di jendela 5x5 dihitung dari dua filter rata-rata (lowpass 1/25, slide 03). Threshold Otsu "
        "membagi piksel blister menjadi halus dan kasar (T = " + bulat(t_sd * SKALA_STD / 255.0) + "). "
        "Piksel kasar dipakai sebagai CONTOH FOIL: kecerahan foil (median) Yf = " + bulat(Yf, 0) +
        ", kroma foil (" + bulat(c1f) + ", " + bulat(c2f) + "). Obat nanti dibandingkan dengan foil ini.",
        g_panas(w, h, sd8, "std lokal (0..40)")))

    # ---------------------------------------------------------------- B4
    batas = BATAS_FOIL * Yf
    g = [(y if y > batas else batas) if m else 255.0 for y, m in zip(c.Y, roi)]
    k = ganjil(0.25 * akar(c.luas_roi))
    k = max(15, min(k, ganjil(min(w, h) / 3.0)))
    th, _ = top_hat(g, w, h, k)
    th8 = skala_8bit([t if m else 0.0 for t, m in zip(th, roi)], 0, SKALA_TOPHAT)
    t_th = max(otsu(histogram(th8, roi)), int(MIN_T_TOPHAT * 255 / SKALA_TOPHAT))
    langkah.append(langkah_baru(
        "Lebih terang dari foil: top-hat", "09 Morfologi (graylevel) + 03 filter min/maks",
        "Opening graylevel = filter minimum lalu maksimum " + str(k) + "x" + str(k) + "; top-hat = f - opening",
        "top-hat = f - max_kxk( min_kxk(f) );   f' = max(f, " + bulat(BATAS_FOIL, 2) + " x Yf) di dalam blister",
        "Opening graylevel dengan structuring element " + str(k) + "x" + str(k) + " (lebih besar dari "
        "obat) menghapus benda TERANG yang lebih kecil dari SE, sehingga hasilnya = perkiraan foil di "
        "sekitar setiap piksel. Top-hat = citra - opening = seberapa terang piksel itu DI ATAS foil di "
        "sekitarnya (tahan terhadap lampu yang tidak rata). Sebelumnya piksel gelap (tulisan, kantong "
        "kosong) dinaikkan ke " + bulat(BATAS_FOIL, 2) + " x Yf supaya tidak menurunkan perkiraan foil, dan "
        "piksel di luar blister diberi 255. Threshold Otsu = " +
        bulat(t_th * SKALA_TOPHAT / 255.0) + " gray level.",
        g_panas(w, h, th8, "top-hat (0..100)")))

    # ---------------------------------------------------------------- B5
    dc = [((a - c1f) ** 2 + (b - c2f) ** 2) ** 0.5 for a, b in zip(c.c1, c.c2)]
    dc8 = skala_8bit(dc, 0, SKALA_KROMA)
    t_dc = max(otsu(histogram(dc8, roi)), int(MIN_T_KROMA * 255 / SKALA_KROMA))
    langkah.append(langkah_baru(
        "Beda warna dari foil", "05 Color Image Processing",
        "Jarak Euclid kroma (c1, c2) setiap piksel ke kroma foil",
        "dK = sqrt( (c1 - " + bulat(c1f) + ")^2 + (c2 - " + bulat(c2f) + ")^2 );  T = " +
        bulat(t_dc * SKALA_KROMA / 255.0),
        "Obat kuning, krem, atau merah belum tentu lebih terang dari foil, tetapi WARNANYA beda. Jarak "
        "Euclid (De, materi 02) di bidang kroma (c1, c2) ke kroma foil menonjolkan obat berwarna. "
        "Threshold Otsu (minimal " + bulat(MIN_T_KROMA, 0) + ") = " + bulat(t_dc * SKALA_KROMA / 255.0) + ".",
        g_panas(w, h, dc8, "beda kroma dari foil (0..60)")))

    # ---------------------------------------------------------------- B6
    kand = []
    for i in range(n):
        if not roi[i] or sd8[i] > t_sd:
            kand.append(0)                         # di luar blister / kasar
        elif th8[i] > t_th:
            kand.append(1)                         # lebih terang dari foil
        elif dc8[i] > t_dc and (c.Y[i] > GELAP_BOLEH * Yf or
                                (c.S[i] > JENUH_GELAP and c.Y[i] > 0.25 * Yf)):
            kand.append(1)                         # warnanya beda dari foil
        else:
            kand.append(0)
    kand = opening(kand, w, h, 3)
    kand = isi_lubang(kand, w, h)
    langkah.append(langkah_baru(
        "Kandidat obat", "02 Enhancement (threshold) + 09 Morfologi",
        "Gabungan threshold (AND / OR), opening 3x3, hole filling",
        "kandidat = ROI AND halus AND ( top-hat > T1  OR  ( dK > T2 AND Y cukup terang ) );  opening 3x3",
        "Piksel menjadi kandidat obat bila di dalam blister, permukaannya halus, dan lebih terang dari "
        "foil ATAU warnanya beda dari foil (piksel berwarna boleh lebih gelap bila jenuh, S > " +
        bulat(JENUH_GELAP, 2) + " dari HSI, untuk belahan kapsul merah). Opening 3x3 membuang bintik, "
        "lalu lubang kecil (huruf timbul di obat) diisi.",
        g_mask(w, h, kand)))
    return kand


# ---------------------------- PENILAIAN REGION ------------------------------
def fitur(c, piksel):
    # Deskriptor sebuah region (materi 09) + pembandingnya: CINCIN di
    # sekeliling region = dilasi besar - dilasi kecil. Obat selalu duduk di
    # dalam kantong blister, jadi cincinnya berisi dinding kantong / foil.
    w, h = c.w, c.h
    d = {"piksel": piksel, "luas": len(piksel)}
    d.update(momen(piksel, w))
    r = LEBAR_CINCIN
    x0, y0, bw, bh, m = kotak_lokal(piksel, r + 1, w, h)
    besar = dilasi(m, bw, bh, 2 * r + 1)
    kecil = dilasi(m, bw, bh, 3)
    cincin = ke_global([q for q in range(bw * bh) if besar[q] and not kecil[q]], x0, y0, bw, w)
    cin_roi = [i for i in cincin if c.roi[i]]
    d["di_roi"] = len(cin_roi) / float(len(cincin) + 1)
    acuan = cin_roi if len(cin_roi) > 10 else cincin
    d["Yin"] = nilai_tengah([c.Y[i] for i in piksel])
    a1 = nilai_tengah([c.c1[i] for i in piksel])
    a2 = nilai_tengah([c.c2[i] for i in piksel])
    k1 = nilai_tengah([c.c1[i] for i in acuan])
    k2 = nilai_tengah([c.c2[i] for i in acuan])
    d["dY"] = d["Yin"] - nilai_tengah([c.Y[i] for i in acuan])      # beda terang dg cincin
    d["dK"] = akar((a1 - k1) ** 2 + (a2 - k2) ** 2)                 # beda kroma dg cincin
    d["dKlatar"] = akar((a1 - c.b1) ** 2 + (a2 - c.b2) ** 2)         # beda kroma dg latar
    d["cinfoil"] = akar((k1 - c.c1f) ** 2 + (k2 - c.c2f) ** 2)       # cincin mirip foil?
    d["S"] = nilai_tengah([c.S[i] for i in piksel])
    d["warna"] = [nilai_tengah([K[i] for i in piksel]) for K in (c.R, c.G, c.B)]
    xs = [i % w for i in piksel]
    ys = [i // w for i in piksel]
    d["x0"], d["x1"], d["y0"], d["y1"] = min(xs), max(xs), min(ys), max(ys)
    return d


def nilai(c, d):
    # aturan keputusan (urut). Hasil "ok" atau alasan penolakan.
    if d["luas"] < c.min_luas:
        return "kecil"
    if d["luas"] > MAKS_LUAS_ROI * c.luas_roi:
        return "besar"
    if d["isi"] < MIN_ISI or d["lonjong"] > MAKS_LONJONG:
        return "bentuk"
    if d["di_roi"] < MIN_DI_ROI or d["cinfoil"] > MAKS_CINCIN_FOIL:
        return "luar"                     # tidak berada di dalam kantong blister
    terang = d["dY"] >= BEDA_TERANG and d["Yin"] >= TERANG_FOIL * c.Yf
    warna = d["dK"] >= BEDA_KROMA and (d["dY"] >= -10 or d["S"] >= JENUH or d["dK"] >= KROMA_KUAT)
    if not (terang or warna):
        return "kontras"                  # tidak menonjol dari kantongnya
    if d["dKlatar"] < MIRIP_LATAR and d["dY"] < BEDA_TERANG:
        return "latar"                    # kantong kosong: yang terlihat meja
    return "ok"


def belah(c, d):
    # REGION SPLITTING (materi 08). Region yang ditolak bisa jadi gabungan
    # obat + foil + pantulan. Region dibelah dua dengan Otsu di dalam region
    # memakai fitur yang PALING memisahkan (eta Otsu terbesar):
    #   Y (kecerahan), kroma, atau tekstur (std lokal).
    piksel = d["piksel"]
    ys = [c.Y[i] for i in piksel]
    m1 = nilai_tengah([c.c1[i] for i in piksel])
    m2 = nilai_tengah([c.c2[i] for i in piksel])
    kd = [akar((c.c1[i] - m1) ** 2 + (c.c2[i] - m2) ** 2) for i in piksel]
    ss = [c.sd[i] for i in piksel]
    pilihan = []
    for nama, nilai_ in (("kecerahan", ys), ("kroma", kd), ("tekstur", ss)):
        v8 = skala_8bit(nilai_, min(nilai_), max(nilai_) + 1)
        T, eta = otsu_eta(v8)
        pilihan.append((eta, nama, T, v8))
    pilihan.sort(key=lambda t: -t[0])
    eta, nama, T, v8 = pilihan[0]
    d["fitur_belah"] = nama
    hasil = []
    for bagian in ([p for p, v in zip(piksel, v8) if v > T], [p for p, v in zip(piksel, v8) if v <= T]):
        if len(bagian) < c.min_luas:
            continue
        x0, y0, bw, bh, m = kotak_lokal(bagian, 2, c.w, c.h)
        m = opening(m, bw, bh, 3)                        # putus jembatan tipis
        m = isi_lubang(m, bw, bh, c.min_luas)            # hanya lubang kecil diisi
        _, daftar = label_region(m, bw, bh, 8)
        for r in daftar:
            if r["luas"] >= c.min_luas:
                hasil.append(ke_global(r["piksel"], x0, y0, bw, c.w))
    return hasil


def evaluasi(c, piksel, tingkat, terima, tolak, catatan):
    d = fitur(c, piksel)
    d["alasan"] = nilai(c, d)
    d["tingkat"] = tingkat
    if d["alasan"] == "ok":
        terima.append(d)
        return
    if d["alasan"] in ("bentuk", "kontras", "latar", "besar") and tingkat < MAKS_TINGKAT \
            and d["luas"] >= 3 * c.min_luas:
        bagian = belah(c, d)
        if bagian and not (len(bagian) == 1 and len(bagian[0]) == d["luas"]):
            catatan[d["fitur_belah"]] = catatan.get(d["fitur_belah"], 0) + 1
            for b in bagian:
                evaluasi(c, b, tingkat + 1, terima, tolak, catatan)
            return
    tolak.append(d)


def berwarna(d):
    # belahan kapsul berwarna (merah, dll.)
    return d["dK"] >= WARNA_KAPSUL or d["S"] >= JENUH


def coba_gabung(c, a, b):
    # Apakah a dan b dua belahan SATU kapsul? Syarat: satu berwarna, satu tidak;
    # luasnya sebanding; jaraknya dekat; dan setelah closing gabungannya satu
    # region berbentuk kapsul (keterisian >= 0.8, kelonjongan <= 4).
    rasio = a["luas"] / float(b["luas"])
    if rasio < 0.4 or rasio > 2.5 or berwarna(a) == berwarna(b):
        return None
    celah = int(1.2 * min(a["b"], b["b"])) + 3
    if a["x0"] > b["x1"] + celah or b["x0"] > a["x1"] + celah or \
            a["y0"] > b["y1"] + celah or b["y0"] > a["y1"] + celah:
        return None
    w = c.w
    x0, y0, bw, bh, m = kotak_lokal(a["piksel"] + b["piksel"], celah + 2, w, c.h)
    ma = [0] * (bw * bh)
    for p in a["piksel"]:
        ma[(p // w - y0) * bw + (p % w - x0)] = 1
    da = dilasi(ma, bw, bh, 2 * celah + 1)
    if not any(da[(p // w - y0) * bw + (p % w - x0)] for p in b["piksel"]):
        return None
    mm = isi_lubang(closing(m, bw, bh, 2 * celah + 1), bw, bh)
    _, daftar = label_region(mm, bw, bh, 8)
    if len(daftar) != 1:
        return None
    px = ke_global(daftar[0]["piksel"], x0, y0, bw, w)
    mo = momen(px, w)
    if mo["isi"] < 0.8 or mo["lonjong"] > 4.0:
        return None
    d = fitur(c, px)
    d["alasan"] = "ok"
    d["kapsul"] = True
    return d


def gabung_kapsul(c, terima, tolak):
    # 1) dua region DITERIMA yang ternyata dua belahan kapsul -> satu
    # 2) belahan berwarna yang diterima sendirian dilengkapi dengan belahan
    #    putihnya yang tadi DITOLAK (tidak cukup terang dari foil)
    n_gabung = 0
    ada = True
    while ada:
        ada = False
        for i in range(len(terima)):
            for j in range(i + 1, len(terima)):
                g = coba_gabung(c, terima[i], terima[j])
                if g is not None:
                    terima = [terima[k] for k in range(len(terima)) if k != i and k != j] + [g]
                    n_gabung += 1
                    ada = True
                    break
            if ada:
                break
    n_lengkap = 0
    for i in range(len(terima)):
        a = terima[i]
        if a.get("kapsul") or not berwarna(a):
            continue
        for j in range(len(tolak)):
            b = tolak[j]
            if b["alasan"] not in ("kontras", "latar", "kecil") or berwarna(b) or b["dY"] < 0:
                continue
            g = coba_gabung(c, a, b)
            if g is not None:
                terima[i] = g
                tolak.pop(j)
                n_lengkap += 1
                break
    return terima, n_gabung, n_lengkap


def ukuran_acuan(daftar):
    # Obat dalam satu foto biasanya sejenis -> ukurannya mirip. Untuk setiap
    # region dihitung kelompok region yang luasnya 0.6..1.67 kali luasnya;
    # kelompok dengan TOTAL luas terbesar menjadi acuan (median luasnya).
    acuan = 0
    skor_terbaik = -1
    for d in daftar:
        kel = [e["luas"] for e in daftar if 0.6 * d["luas"] <= e["luas"] <= 1.67 * d["luas"]]
        if sum(kel) > skor_terbaik:
            skor_terbaik = sum(kel)
            acuan = nilai_tengah(kel)
    return acuan


def nama_bentuk(d):
    # nama bentuk dari deskriptor region (materi 09)
    if d.get("kapsul"):
        return "kapsul"
    if d["lonjong"] < LONJONG_BULAT:
        return "bulat"
    if berwarna(d) and d["dK"] >= KROMA_KUAT:
        return "kapsul"
    return "kaplet / oval"


def deskripsi_bentuk(c, d):
    # chain code 8 arah -> keliling P; pusat (centroid); jari-jari lingkaran
    # penutup = jarak Euclid terjauh dari pusat ke titik batas;
    # kebundaran 4 pi A / P^2 (Gonzalez, 1 = lingkaran sempurna)
    kode, titik = chain_code(d["piksel"], c.w)
    d["chain"] = kode
    d["titik_batas"] = titik
    d["keliling"] = keliling(kode)
    d["kebundaran"] = 4 * PI * d["luas"] / (d["keliling"] ** 2) if d["keliling"] > 0 else 0
    r = 0.0
    for (x, y) in titik:
        jarak = akar((x - d["cx"]) ** 2 + (y - d["cy"]) ** 2)
        if jarak > r:
            r = jarak
    d["r"] = r + 2.0
    # arah sumbu panjang elips setara (momen orde 2):
    #   theta = 1/2 atan2( 2 mu11, mu20 - mu02 )   (derajat, sumbu y ke bawah)
    d["sudut"] = 0.5 * arctan2(2 * d["m11"], d["m20"] - d["m02"]) * 180.0 / PI
    d["bentuk"] = nama_bentuk(d)
    return d


# ================================ ANALISIS ===================================
def analisis_data(lebar0, tinggi0, data):
    langkah = []
    acuan, fa, kotak = tahap_a(lebar0, tinggi0, data, langkah)
    c = tahap_b_awal(lebar0, tinggi0, data, acuan, fa, kotak, langkah)
    kand = tahap_b_peta(c, langkah)

    # ---------------------------------------------------------------- B7
    _, daftar = label_region(kand, c.w, c.h, 8)
    terima = []
    tolak = []
    catatan_belah = {}
    n_awal = 0
    for d in daftar:
        if d["luas"] >= c.min_luas:
            n_awal += 1
            evaluasi(c, d["piksel"], 0, terima, tolak, catatan_belah)
    hitung = {}
    for d in tolak:
        hitung[d["alasan"]] = hitung.get(d["alasan"], 0) + 1
    teks_tolak = ", ".join(k + " " + str(v) for k, v in sorted(hitung.items())) or "-"
    teks_belah = ", ".join(k + " " + str(v) + "x" for k, v in sorted(catatan_belah.items())) or "tidak ada"
    langkah.append(langkah_baru(
        "Pelabelan, penilaian & region splitting", "02 Konektivitas + 08 Segmentasi + 09 Deskripsi",
        "Label N8, momen (luas, keterisian, kelonjongan), cincin pembanding, aturan; region splitting Otsu",
        "cincin = dilasi(A, " + str(2 * LEBAR_CINCIN + 1) + ") - dilasi(A, 3);  obat putih: dY >= " +
        str(BEDA_TERANG) + " dan Y >= " + bulat(TERANG_FOIL, 2) + " Yf;  obat berwarna: dK >= " +
        str(BEDA_KROMA),
        "Kandidat dilabeli dengan konektivitas N8: " + str(n_awal) + " region. Setiap region diberi "
        "deskriptor: luas A, keterisian A / luas elips setara dan kelonjongan (momen orde 2, Gonzalez), "
        "lalu dibandingkan dengan CINCIN di sekelilingnya (dinding kantong / foil). Aturan: luas wajar "
        "(<= " + bulat(MAKS_LUAS_ROI * 100, 0) + "% blister), bentuk rapi (keterisian >= " +
        bulat(MIN_ISI, 2) + ", kelonjongan <= " + bulat(MAKS_LONJONG, 1) + "), cincinnya di dalam blister dan "
        "berwarna foil, obat putih lebih terang >= " + str(BEDA_TERANG) + " dari cincin DAN >= " +
        bulat(TERANG_FOIL, 2) + " x kecerahan foil, obat berwarna beda kroma >= " + str(BEDA_KROMA) +
        ", dan tidak berwarna latar (kantong kosong yang menampakkan meja). Region yang gagal dibelah "
        "dua dengan Otsu di dalam region pada fitur yang paling memisahkan (region splitting, slide 08): " +
        teks_belah + ". Diterima " + str(len(terima)) + " region (hijau), ditolak: " + teks_tolak + " (merah).",
        g_overlay(c, [(tolak, (255, 70, 70), False), (terima, (60, 255, 100), True)])))

    # ---------------------------------------------------------------- B8
    terima, n_gabung, n_lengkap = gabung_kapsul(c, terima, tolak)
    kapsul = [d for d in terima if d.get("kapsul")]
    langkah.append(langkah_baru(
        "Gabung belahan kapsul", "09 Morfologi (closing) + 05 Warna",
        "Closing menyatukan dua belahan beda warna; region growing ke belahan putih",
        "syarat: satu belahan berwarna (dK >= " + str(WARNA_KAPSUL) + " atau S >= " + bulat(JENUH, 2) +
        "), luas 0.4..2.5x, closing -> 1 region, keterisian >= 0.8, kelonjongan <= 4",
        "Kapsul punya dua belahan dengan warna berbeda (mis. merah | putih) dan di antaranya ada garis "
        "sambungan, sehingga terdeteksi sebagai DUA region. Dua region berdekatan yang salah satunya "
        "berwarna disatukan dengan closing (dilasi lalu erosi, slide 09: 'menggabungkan 2 objek yang salah "
        "menjadi sebuah objek yang benar'); hasilnya dipakai hanya bila berbentuk kapsul. Belahan berwarna "
        "yang diterima sendirian dilengkapi dengan belahan putihnya yang tadi ditolak. Disatukan: " +
        str(n_gabung) + ", dilengkapi: " + str(n_lengkap) + ".",
        g_overlay(c, [(kapsul, (255, 200, 40), True), ([d for d in terima if not d.get("kapsul")], (60, 255, 100), True)]),
        n_gabung + n_lengkap > 0))

    # ---------------------------------------------------------------- B9
    obat = []
    acuan_luas = ukuran_acuan(terima) if terima else 0
    for d in terima:
        if d["luas"] < KECIL_ACUAN * acuan_luas:
            d["alasan"] = "kecil"
            tolak.append(d)
        elif d["luas"] > BESAR_ACUAN * acuan_luas and not d.get("kapsul"):
            d["alasan"] = "besar"
            tolak.append(d)
        else:
            obat.append(d)
    n_ukuran = len(terima) - len(obat)
    id_obat = set(id(d) for d in obat)
    langkah.append(langkah_baru(
        "Konsistensi ukuran", "09 Representation & Description",
        "Luas acuan = median kelompok luas yang paling dominan",
        "acuan = " + str(acuan_luas) + " px;  ditolak bila A < " + bulat(KECIL_ACUAN, 1) + " x acuan atau A > " +
        bulat(BESAR_ACUAN, 1) + " x acuan",
        "Obat dalam satu foto biasanya sejenis sehingga luasnya mirip. Untuk setiap region dicari region "
        "lain yang luasnya 0.6..1.67 kali; kelompok dengan total luas terbesar dipakai sebagai acuan "
        "(median luasnya = " + str(acuan_luas) + " piksel). Serpihan (< " + bulat(KECIL_ACUAN, 1) +
        "x) dan benda yang terlalu besar (> " + bulat(BESAR_ACUAN, 1) + "x) ditolak: " + str(n_ukuran) + " region.",
        g_overlay(c, [(obat, (60, 255, 100), True), ([d for d in terima if id(d) not in id_obat], (255, 70, 70), False)]),
        n_ukuran > 0))

    # ---------------------------------------------------------------- B10
    for d in obat:
        deskripsi_bentuk(c, d)
    obat.sort(key=lambda d: (int(d["cy"] / max(1.0, 2 * d["b"])), d["cx"]))   # urut baris lalu kolom
    R = [int(v * 0.5) for v in c.R]
    G = [int(v * 0.5) for v in c.G]
    B = [int(v * 0.5) for v in c.B]
    for d in obat:
        for (x, y) in d["titik_batas"]:
            i = y * c.w + x
            R[i], G[i], B[i] = 60, 255, 100
        x, y = d["titik_batas"][0]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if 0 <= x + dx < c.w and 0 <= y + dy < c.h:
                    i = (y + dy) * c.w + x + dx
                    R[i], G[i], B[i] = 255, 230, 0
    teks = []
    for k, d in enumerate(obat):
        teks.append("#" + str(k + 1) + " " + d["bentuk"] + ": A=" + str(d["luas"]) + ", P=" + bulat(d["keliling"], 0) +
                    ", 4piA/P^2=" + bulat(d["kebundaran"], 2) + ", lonjong=" + bulat(d["lonjong"], 2))
    langkah.append(langkah_baru(
        "Chain code & deskripsi bentuk", "09 Representation & Description",
        "Chain code 8 arah, keliling, centroid, kelonjongan, kebundaran",
        "P = #kode genap + sqrt(2) x #kode ganjil;  pusat = rata-rata (x, y);  4 pi A / P^2;  "
        "bulat bila lonjong < " + bulat(LONJONG_BULAT, 2),
        "Batas luar setiap obat ditelusuri dengan chain code 8 arah (3 2 1 / 4 x 0 / 5 6 7, slide 09; "
        "titik kuning = titik awal). Dari chain code dihitung keliling P (kode genap = 1 langkah, kode "
        "ganjil = sqrt 2). Nama bentuk: kelonjongan < " + bulat(LONJONG_BULAT, 2) + " -> bulat, region "
        "gabungan dua warna -> kapsul, selain itu kaplet / oval. " + ("; ".join(teks) + "." if teks else
                                                                   "Tidak ada obat."),
        g_rgb(c.w, c.h, R, G, B), len(obat) > 0))

    # ---------------------------------------------------------------- B11
    per_bentuk = {}
    for d in obat:
        per_bentuk[d["bentuk"]] = per_bentuk.get(d["bentuk"], 0) + 1
    teks_bentuk = ", ".join(k + " " + str(v) for k, v in per_bentuk.items())
    langkah.append(langkah_baru(
        "Hasil: sisa obat", "09 Representation & Description",
        "Elips setara dari momen: pusat, sumbu a & b, arah theta",
        "pusat = (rata x, rata y);  a = 2 sqrt(lambda1), b = 2 sqrt(lambda2);  theta = 1/2 atan2(2 mu11, mu20 - mu02)",
        "Setiap obat ditandai ELIPS SETARA dari momen orde 2 region-nya (Gonzalez): pusat = centroid, "
        "setengah sumbu a dan b dari nilai eigen lambda1, lambda2, dan arah sumbu panjang theta. Elips ini "
        "mengikuti bentuk obat (bulat, kaplet, kapsul) dan kemiringannya. SISA OBAT = " + str(len(obat)) +
        (" (" + teks_bentuk + ")." if teks_bentuk else "."),
        {"jenis": "hasil"}))

    hasil_obat = [{"cx": d["cx"], "cy": d["cy"], "r": d["r"], "a": d["a"], "b": d["b"], "sudut": d["sudut"],
                   "bentuk": d["bentuk"], "luas": d["luas"],
                   "keliling": d["keliling"], "kebundaran": d["kebundaran"], "lonjong": d["lonjong"],
                   "x0": d["x0"], "y0": d["y0"], "x1": d["x1"], "y1": d["y1"]} for d in obat]
    hasil_tolak = []
    for d in tolak:
        if d["luas"] >= 0.5 * max(acuan_luas, c.min_luas):
            hasil_tolak.append({"cx": d["cx"], "cy": d["cy"], "alasan": d["alasan"],
                                "r": max(d["x1"] - d["x0"], d["y1"] - d["y0"]) / 2.0 + 2})
    return {"jumlah": len(obat), "obat": hasil_obat, "ditolak": hasil_tolak, "langkah": langkah,
            "foto": g_rgb(c.w, c.h, c.R, c.G, c.B), "kotak": c.kotak, "faktor": c.f,
            "acuan_luas": acuan_luas, "Yf": c.Yf,
            "_obat_lengkap": obat, "_tolak_lengkap": tolak}
