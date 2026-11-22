# =============================================================================
# deteksi.py  -  ALUR DETEKSI SISA OBAT DALAM BLISTER (TANPA MACHINE LEARNING)
#
# Semua pengolahan citra memakai rumus manual dari citra.py. Satu-satunya
# library di file ini adalah PIL di fungsi baca_foto(), dan itu HANYA untuk
# membuka file JPEG/PNG menjadi deretan byte R,G,B (input). Setelah itu tidak
# ada library lagi. Tidak ada pelatihan / dataset / model: semua keputusan
# berupa aturan (threshold) yang bisa dijelaskan dengan materi kuliah.
#
# ALUR
#   A. Cari blister kasar (foto diperkecil, sisi terpanjang 400 px)
#        foil = kebiruan (Cb dari YCbCr) [+ bertekstur (energi highpass)]
#        -> Otsu -> closing -> isi lubang -> opening -> komponen terbesar
#   B. Potong blister dari foto ASLI lalu perkecil (sisi 480 px) -> obat cukup
#      besar untuk dianalisis.
#   C. Perbaikan citra otomatis dari statistik histogram:
#        salt & pepper -> median, noise -> lowpass, gelap -> gamma,
#        kontras rendah -> contrast stretching
#   D. Mask blister di potongan (Cb/tekstur, digabung dengan ROI tahap A),
#      bungkus oktagon. Warna pembanding: MEJA (median di luar blister) dan
#      FOIL LOKAL (median per blok piksel foil).
#   E. Calon obat:
#        biji  = halus (std lokal kecil) DAN mirip obat
#                (warna jauh dari foil lokal, bukan warna meja, dan
#                 [kurang biru dari foil] atau [jauh lebih terang] atau
#                 [warna mencolok])
#        region growing dari biji -> morfologi -> label komponen terhubung
#   F. Setiap region dinilai dengan deskriptor (momen, warna, cincin):
#        luar / kosong / di meja / kasar / foil / plastik / pantulan / bentuk
#      region yang gagal karena bentuk/posisi dipecah dengan opening
#      bertingkat lalu dinilai ulang. Terakhir aturan ukuran:
#        < 0.30 x median -> serpihan,  < 0.70 x median -> potongan obat
# =============================================================================

import time

from citra import *


# ------------------------------ PARAMETER ------------------------------------
# tahap A-B: mencari & memotong blister
SISI_KASAR = 400          # tahap A: sisi terpanjang foto utuh
SISI_DETAIL = 480         # tahap B: sisi terpanjang potongan blister
MARGIN_POTONG = 0.20      # potongan diperlebar 20% dari ukuran blister
JENDELA_TEKSTUR_A = 9     # jendela energi tekstur tahap A
JENDELA_TEKSTUR_D = 7     # jendela energi tekstur tahap D
BATAS_ETA_WARNA = 0.85    # keterpisahan Otsu Cb >= 0.85 -> cukup ciri warna
BATAS_SATURASI = 0.30     # saturasi >= 0.30 dianggap pasti berwarna (bukan foil)
MIN_LUAS_BLISTER = 0.10   # mask blister < 10% potongan -> dianggap gagal

# tahap C: perbaikan citra
BATAS_SALT = 0.004        # > 0.4% piksel ekstrem terisolasi -> median filter
BATAS_NOISE = 4.0         # residu noise di daerah datar > 4 -> lowpass
BATAS_GELAP = 90          # rata-rata keabuan < 90 -> gamma (terangkan)
BATAS_KONTRAS = 45        # std keabuan < 45 -> contrast stretching
BATAS_RENTANG = 60        # rentang p1..p99 < 60 -> histogram equalization

# tahap E: calon obat (per piksel)
JENDELA_STD = 7           # jendela std lokal (kehalusan)
BATAS_HALUS = 12          # std lokal < 12 -> halus
BLOK_FOIL = 24            # ukuran blok median warna foil lokal
BEDA_FOIL = 40            # biji: jarak warna ke foil lokal > 40
BEDA_FOIL_TUMBUH = 25     # region growing: jarak warna ke foil lokal > 25
BEDA_MEJA = 45            # jarak warna ke meja < 45 -> warna meja
SELISIH_BIRU = 0.01       # obat: b = B/(R+G+B) minimal 0.01 di bawah foil
RASIO_TERANG = 1.25       # atau >= 1.25 x lebih terang dari foil
BATAS_MENCOLOK = 0.45     # atau saturasi > 0.45 (obat berwarna, mis. merah)
RASIO_GELAP = 0.85        # obat tidak lebih gelap dari 0.85 x foil sekitarnya
LANGKAH_TUMBUH = 4        # region growing maksimal 4 piksel ke luar

# tahap F: penilaian region
MIN_LUAS_RELATIF = 0.002  # region < 0.2% luas blister diabaikan
CLOSING_DALAM = 41        # closing penutup coakan mask blister
MIN_DALAM = 0.60          # >= 60% region harus di dalam blister
BEDA_KROMATIS_MEJA = 0.03 # kromatisitas (r,g) < 0.03 dari meja -> kantong kosong
MAKS_CINCIN_MEJA = 0.35   # > 35% cincin berwarna meja -> pantulan di meja
MIN_BIJI = 0.35           # >= 35% piksel region harus biji (halus & mirip obat)
BEDA_FOIL_REGION = 50     # rata-rata jarak warna region ke foil lokal > 50
SELISIH_BIRU_REGION = 0.02  # rata-rata b region minimal 0.02 di bawah foil ...
RASIO_TERANG_REGION = 1.33  # ... kecuali region >= 1.33 x lebih terang dari foil
BATAS_KILAU = 245         # min(R,G,B) >= 245 = putih jenuh (pantulan)
MAKS_KILAU = 0.85         # >= 85% region putih jenuh -> pantulan
MIN_ISI = 0.70            # luas / luas elips momen (bentuk kompak)
MAKS_LONJONG = 4.5        # sumbu panjang / sumbu pendek
PECAH_BERTINGKAT = (9, 13, 17)  # opening bertingkat untuk memecah region gagal
BATAS_SERPIHAN = 0.30     # luas < 0.30 x median obat -> serpihan
BATAS_POTONGAN = 0.70     # luas < 0.70 x median obat -> potongan obat


# ------------------------------ INPUT ----------------------------------------
def baca_foto(path):
    # SATU-SATUNYA pemakaian library untuk gambar: membuka file (input).
    # Hasilnya deretan byte R,G,B; semua proses berikutnya manual.
    from PIL import Image
    gambar = Image.open(path)
    lebar, tinggi = gambar.size
    if gambar.mode == "RGB":
        return lebar, tinggi, gambar.tobytes()
    if gambar.mode in ("L", "RGBA"):
        # disusun manual: abu-abu -> R=G=B, RGBA -> buang kanal alpha
        mentah = gambar.tobytes()
        if gambar.mode == "L":
            data = bytearray(lebar * tinggi * 3)
            data[0::3] = mentah
            data[1::3] = mentah
            data[2::3] = mentah
        else:
            data = bytearray(lebar * tinggi * 3)
            data[0::3] = mentah[0::4]
            data[1::3] = mentah[1::4]
            data[2::3] = mentah[2::4]
        return lebar, tinggi, bytes(data)
    # format lain (mis. palet) -> minta PIL membukanya sebagai RGB
    return lebar, tinggi, gambar.convert("RGB").tobytes()


# ======================= A. CARI BLISTER (KASAR) =============================
def peta_foil(R, G, B, lebar, tinggi, k):
    """Skor 'seberapa mirip foil' setiap piksel (0..255).

    1. Highpass (03 Spatial Filtering):  hp = |f - lowpass_3x3(f)|
       Pola embos foil = perubahan cepat -> hp besar.
    2. Energi tekstur = rata-rata hp di jendela k x k, dibagi kecerahan
       lokal (+20) supaya foil di foto gelap tetap terdeteksi.
    3. Dikalikan (1 - S/0.30): foil itu perak/abu-abu (saturasi kecil),
       meja kayu berwarna (saturasi besar).
    """
    abu = ke_abu(R, G, B)
    lowpass = filter_rata(abu, lebar, tinggi, 3)
    hp = [abs(abu[i] - lowpass[i]) for i in range(len(abu))]
    energi = filter_rata(hp, lebar, tinggi, k)
    terang = filter_rata(abu, lebar, tinggi, k)
    sat = filter_rata(saturasi_hsi(R, G, B), lebar, tinggi, 5)
    skor = [0.0] * len(abu)
    for i in range(len(abu)):
        tidak_berwarna = 1 - sat[i] / BATAS_SATURASI
        if tidak_berwarna < 0:
            tidak_berwarna = 0
        skor[i] = energi[i] / (terang[i] + 20) * tidak_berwarna
    return normalisasi(skor)


def segmentasi_blister(R, G, B, lebar, tinggi, k_tekstur):
    """Mask blister. Dua ciri foil:
      (1) WARNA: kebiruan -> Cb tinggi (meja kayu kekuningan -> Cb rendah)
      (2) TEKSTUR: pola embos -> energi highpass tinggi (peta_foil)
    Threshold keduanya dengan Otsu. Keterpisahan eta(Cb) dipakai memilih:
      eta >= 0.85 -> histogram Cb jelas dua puncak (meja polos, cahaya baik)
                     -> cukup ciri warna
      eta <  0.85 -> latar ramai / gelap -> foil = Cb tinggi DAN bertekstur
    """
    n = lebar * tinggi
    cb = filter_rata(ke_cb(R, G, B), lebar, tinggi, 5)
    hist_cb = histogram(cb)
    batas_cb = otsu(hist_cb)
    eta = keterpisahan_otsu(hist_cb, batas_cb)
    skor = peta_foil(R, G, B, lebar, tinggi, k_tekstur)
    batas_tekstur = otsu(histogram(skor))
    tekstur = [1 if v > batas_tekstur else 0 for v in skor]
    if eta >= BATAS_ETA_WARNA:
        mode = "warna (Cb)"
        mask = [1 if v > batas_cb else 0 for v in cb]
    else:
        mode = "warna (Cb) + tekstur"
        mask = [1 if (cb[i] > batas_cb and tekstur[i]) else 0 for i in range(n)]
    mask = closing(mask, lebar, tinggi, 9)
    mask = isi_lubang(mask, lebar, tinggi)
    mask = opening(mask, lebar, tinggi, 9)
    mask = komponen_terbesar(mask, lebar, tinggi)
    return {"mask": mask, "mode": mode, "eta": eta, "batas_cb": batas_cb, "cb": cb,
            "skor_tekstur": skor, "tekstur": tekstur, "batas_tekstur": batas_tekstur}


def cari_blister_kasar(data, W, H):
    R, G, B, w, h, f = sampling_ulang(data, W, H, (0, 0, W, H), SISI_KASAR, rata=False)
    seg = segmentasi_blister(R, G, B, w, h, JENDELA_TEKSTUR_A)
    roi = seg["mask"]
    x0, y0, x1, y1 = kotak_pembatas(roi, w, h)
    m = int(MARGIN_POTONG * max(x1 - x0, y1 - y0))
    x0 = max(0, x0 - m)
    y0 = max(0, y0 - m)
    x1 = min(w, x1 + m)
    y1 = min(h, y1 + m)
    kotak = (int(x0 * f), int(y0 * f), min(W, int(x1 * f)), min(H, int(y1 * f)))
    return {"kecil": (R, G, B, w, h), "roi_kasar": roi, "mode": seg["mode"], "eta": seg["eta"],
            "kotak_kecil": (x0, y0, x1, y1), "kotak": kotak}


# ======================= C. PERBAIKAN CITRA OTOMATIS =========================
def ukur_salt_pepper(abu, lebar, tinggi):
    # Noise salt & pepper = piksel EKSTREM (<=10 atau >=245) yang
    # TERISOLASI: beda > 60 dari rata-rata 4 tetangganya.
    n = 0
    for y in range(1, tinggi - 1):
        for x in range(1, lebar - 1):
            i = y * lebar + x
            v = abu[i]
            if v <= 10 or v >= 245:
                tetangga = (abu[i - 1] + abu[i + 1] + abu[i - lebar] + abu[i + lebar]) / 4
                if abs(v - tetangga) > 60:
                    n += 1
    return n / len(abu)


def ukur_noise(abu, lebar, tinggi):
    # residu = f - lowpass(f); noise = median |residu| dari blok 8x8
    # yang paling datar (persentil 25%) supaya tepi objek tidak ikut
    lowpass = filter_rata(abu, lebar, tinggi, 3)
    nilai_blok = []
    for by in range(tinggi // 8):
        for bx in range(lebar // 8):
            s = 0.0
            for y in range(by * 8, by * 8 + 8):
                for x in range(bx * 8, bx * 8 + 8):
                    i = y * lebar + x
                    s += abs(abu[i] - lowpass[i])
            nilai_blok.append(s / 64)
    if not nilai_blok:
        return 0.0
    nilai_blok = sorted_manual(nilai_blok)
    return nilai_blok[len(nilai_blok) // 4]


def sorted_manual(daftar):
    # merge sort manual (untuk daftar panjang, insertion sort terlalu lambat)
    if len(daftar) <= 1:
        return list(daftar)
    tengah = len(daftar) // 2
    kiri = sorted_manual(daftar[:tengah])
    kanan = sorted_manual(daftar[tengah:])
    hasil = []
    i = 0
    j = 0
    while i < len(kiri) and j < len(kanan):
        if kiri[i] <= kanan[j]:
            hasil.append(kiri[i])
            i += 1
        else:
            hasil.append(kanan[j])
            j += 1
    hasil.extend(kiri[i:])
    hasil.extend(kanan[j:])
    return hasil


def perbaiki_citra(R, G, B, lebar, tinggi):
    langkah = []
    abu = ke_abu(R, G, B)

    # 1. salt & pepper -> median filter 3x3 (per kanal)
    salt = ukur_salt_pepper(abu, lebar, tinggi)
    if salt > BATAS_SALT:
        R = filter_median(R, lebar, tinggi)
        G = filter_median(G, lebar, tinggi)
        B = filter_median(B, lebar, tinggi)
        langkah.append(("Median filter 3x3", True, "piksel ekstrem %.2f%% > %.2f%%" % (salt * 100, BATAS_SALT * 100)))
    else:
        langkah.append(("Median filter 3x3", False, "salt & pepper %.2f%% (kecil)" % (salt * 100)))
    abu = ke_abu(R, G, B)

    # 2. gaussian noise -> lowpass 3x3 (bobot 1-2-1)
    noise = ukur_noise(abu, lebar, tinggi)
    if noise > BATAS_NOISE:
        R = konvolusi(R, lebar, tinggi, KERNEL_LOWPASS)
        G = konvolusi(G, lebar, tinggi, KERNEL_LOWPASS)
        B = konvolusi(B, lebar, tinggi, KERNEL_LOWPASS)
        langkah.append(("Lowpass filter 3x3", True, "noise %.1f > %.1f" % (noise, BATAS_NOISE)))
    else:
        langkah.append(("Lowpass filter 3x3", False, "noise %.1f (kecil)" % noise))
    abu = ke_abu(R, G, B)
    asli_bersih = (R, G, B)

    # 3. terlalu gelap -> gamma correction (gamma < 1 menerangkan)
    hist = histogram(abu)
    rata, std = statistik(hist)
    if rata < BATAS_GELAP:
        gamma = 0.5 if rata < 60 else 0.7
        lut = lut_gamma(gamma)
        R = pakai_lut(R, lut)
        G = pakai_lut(G, lut)
        B = pakai_lut(B, lut)
        langkah.append(("Gamma correction", True, "rata-rata %.0f < %d -> gamma %.1f" % (rata, BATAS_GELAP, gamma)))
    else:
        langkah.append(("Gamma correction", False, "rata-rata %.0f (cukup terang)" % rata))
    abu = ke_abu(R, G, B)

    # 4. kontras rendah -> histogram equalization (histogram sangat sempit)
    #    atau contrast stretching (persentil 1% .. 99%)
    hist = histogram(abu)
    rata, std = statistik(hist)
    rentang = persentil(hist, 0.99) - persentil(hist, 0.01)
    if rentang < BATAS_RENTANG:
        lut = lut_equalization(hist)
        R = pakai_lut(R, lut)
        G = pakai_lut(G, lut)
        B = pakai_lut(B, lut)
        langkah.append(("Histogram equalization", True, "rentang p1-p99 = %d < %d" % (rentang, BATAS_RENTANG)))
        langkah.append(("Contrast stretching", False, "diganti equalization"))
    elif std < BATAS_KONTRAS:
        bawah = persentil(hist, 0.01)
        atas = persentil(hist, 0.99)
        lut = lut_stretching(bawah, atas)
        R = pakai_lut(R, lut)
        G = pakai_lut(G, lut)
        B = pakai_lut(B, lut)
        langkah.append(("Contrast stretching", True, "std %.0f < %d -> [%d..%d] ke [0..255]" % (std, BATAS_KONTRAS, bawah, atas)))
    else:
        langkah.append(("Contrast stretching", False, "std %.0f (kontras cukup)" % std))
    if rentang >= BATAS_RENTANG:
        langkah.insert(len(langkah) - 1, ("Histogram equalization", False,
                                          "rentang p1-p99 = %d (cukup lebar)" % rentang))
    return R, G, B, asli_bersih, langkah


# ===================== D-F. SEGMENTASI & PENILAIAN OBAT ======================
def rata_warna(R, G, B, piksel):
    n = len(piksel)
    sr = 0.0
    sg = 0.0
    sb = 0.0
    for i in piksel:
        sr += R[i]
        sg += G[i]
        sb += B[i]
    return sr / n, sg / n, sb / n


def jarak_kromatis(a, b):
    # jarak di bidang kromatisitas (r, g):  r = R/(R+G+B), g = G/(R+G+B)
    ja = a[0] + a[1] + a[2] + 1
    jb = b[0] + b[1] + b[2] + 1
    return ((a[0] / ja - b[0] / jb) ** 2 + (a[1] / ja - b[1] / jb) ** 2) ** 0.5


def jarak3(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def pindahkan_roi(kasar, W, H, lebar, tinggi, f):
    # memindahkan mask ROI tahap A (foto kecil) ke koordinat potongan tahap B
    # (sampling ulang tetangga terdekat)
    _, _, _, wk, hk = kasar["kecil"]
    fk = max(W, H) / max(wk, hk)
    kotak = kasar["kotak"]
    roi = kasar["roi_kasar"]
    hasil = [0] * (lebar * tinggi)
    for y in range(tinggi):
        yk = int((kotak[1] + y * f) / fk)
        if yk > hk - 1:
            yk = hk - 1
        for x in range(lebar):
            xk = int((kotak[0] + x * f) / fk)
            if xk > wk - 1:
                xk = wk - 1
            hasil[y * lebar + x] = roi[yk * wk + xk]
    return hasil


def cincin_meja(d, jarak_meja, lebar, tinggi):
    # cincin = dilasi(region, 11) - dilasi(region, 3)  (pita ~4 px di luar region)
    # hasil = bagian cincin yang warnanya mirip meja
    x0 = lebar
    y0 = tinggi
    x1 = 0
    y1 = 0
    for i in d["piksel"]:
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
    r = 7
    x0 = max(0, x0 - r)
    y0 = max(0, y0 - r)
    x1 = min(lebar - 1, x1 + r)
    y1 = min(tinggi - 1, y1 + r)
    lw = x1 - x0 + 1
    lh = y1 - y0 + 1
    m = [0] * (lw * lh)
    for i in d["piksel"]:
        m[(i // lebar - y0) * lw + (i % lebar - x0)] = 1
    luar = dilasi(m, lw, lh, 11)
    dalam = dilasi(m, lw, lh, 3)
    n_cincin = 0
    n_meja = 0
    for yy in range(lh):
        for xx in range(lw):
            j = yy * lw + xx
            if luar[j] and not dalam[j]:
                n_cincin += 1
                if jarak_meja[(yy + y0) * lebar + xx + x0] < BEDA_MEJA:
                    n_meja += 1
    if n_cincin == 0:
        return 0.0
    return n_meja / n_cincin


def analisis(path, catat=None, progres=None):
    """Menjalankan seluruh alur. catat(teks): log detail region (opsional).
    progres(teks): laporan tahap yang sedang dikerjakan (untuk tampilan)."""
    def lapor(teks):
        if progres is not None:
            progres(teks)

    mulai = time.time()
    lapor("Membaca foto...")
    W, H, data = baca_foto(path)

    # ---------------- A. blister kasar ----------------
    lapor("A. Mencari blister (Cb + tekstur, Otsu, morfologi)...")
    kasar = cari_blister_kasar(data, W, H)

    # ---------------- B. potong dari foto asli ---------
    lapor("B. Memotong blister dari foto asli...")
    R, G, B, w, h, f = sampling_ulang(data, W, H, kasar["kotak"], SISI_DETAIL, rata=False)
    n = w * h
    potongan_asli = (R, G, B)

    # ---------------- C. perbaikan citra --------------
    lapor("C. Perbaikan citra otomatis (histogram)...")
    R, G, B, bersih, langkah = perbaiki_citra(R, G, B, w, h)

    # ---------------- D. mask blister & warna pembanding
    lapor("D. Segmentasi blister & warna pembanding...")
    seg = segmentasi_blister(R, G, B, w, h, JENDELA_TEKSTUR_D)
    skor = seg["skor_tekstur"]
    batas_foil = seg["batas_tekstur"]
    foil = seg["tekstur"]
    # mask blister = GABUNGAN (union) mask tahap D dan ROI tahap A yang
    # dipindah ke koordinat potongan, lalu lubangnya diisi. Keduanya saling
    # menambal bila salah satu bolong (mis. foto gelap / buram).
    roi_a = pindahkan_roi(kasar, W, H, w, h, f)
    if sum(seg["mask"]) < MIN_LUAS_BLISTER * n:
        blister_asli = roi_a
        sumber_blister = "ROI tahap A (tahap D gagal)"
    else:
        blister_asli = [1 if (a_ or d_) else 0 for a_, d_ in zip(seg["mask"], roi_a)]
        sumber_blister = "tahap D (" + seg["mode"] + ") + ROI tahap A"
    blister_asli = komponen_terbesar(isi_lubang(blister_asli, w, h), w, h)
    blister = bungkus_oktagon(blister_asli, w, h)
    # daerah "di dalam blister" untuk menilai region: closing besar menutup
    # coakan di pinggir mask (obat di tepi blister sering membuat coakan)
    sekitar_blister = dilasi(closing(blister_asli, w, h, CLOSING_DALAM), w, h, 7)
    luas_blister = sum(blister)

    luar = [1 - v for v in dilasi(blister, w, h, 9)]
    warna_meja = None
    if sum(luar) > 0.05 * n:
        warna_meja = warna_median(R, G, B, luar)
    warna_foil = warna_median(R, G, B, [foil[i] * blister[i] for i in range(n)])

    # ---------------- E. calon obat -------------------
    lapor("E. Mencari calon obat + region growing...")
    abu = ke_abu(R, G, B)
    rb, gb, bb = bersih
    # saturasi diukur SEBELUM gamma/stretching (stretching membuat piksel
    # gelap tampak sangat berwarna); V = max(R,G,B) dari model HSV
    sat = saturasi_hsi(rb, gb, bb)
    nilai_v = [r if (r >= g and r >= b_) else (g if g >= b_ else b_) for r, g, b_ in zip(R, G, B)]
    std = std_lokal(filter_rata(abu, w, h, 3), w, h, JENDELA_STD)

    # WARNA FOIL LOKAL = median piksel foil per blok (per kanal R, G, B, abu).
    # Median tidak terpengaruh kilau di tepi kantong; bayangan / gradasi
    # cahaya di atas foil ikut terkompensasi karena dihitung per daerah.
    mask_foil = [foil[i] * blister[i] for i in range(n)]
    foil_r, foil_g, foil_b, foil_lokal = [median_per_blok(kanal, mask_foil, w, h, BLOK_FOIL)
                                          for kanal in (R, G, B, abu)]
    # acuan "jauh lebih terang": maks(foil lokal, median foil seluruh blister)
    # -> foil di daerah bayangan tidak tiba-tiba tampak "terang"
    foil_global = persentil(histogram(abu, mask_foil), 0.5)
    acuan_terang = [v if v > foil_global else foil_global for v in foil_lokal]
    jarak_foil = [((R[i] - foil_r[i]) ** 2 + (G[i] - foil_g[i]) ** 2 + (B[i] - foil_b[i]) ** 2) ** 0.5
                  for i in range(n)]
    # Kromatisitas biru (normalized RGB):  b = B / (R + G + B)
    # Foil & plastik kantong memantulkan cahaya kebiruan yang sama (b tinggi),
    # permukaan obat (putih / krem / berwarna) selalu kurang biru dari foil.
    biru = [b_ / (r + g + b_ + 1) for r, g, b_ in zip(R, G, B)]
    biru_foil = [b_ / (r + g + b_ + 1) for r, g, b_ in zip(foil_r, foil_g, foil_b)]
    jarak_meja = jarak_warna(R, G, B, warna_meja) if warna_meja else [999.0] * n

    def mirip_obat(i, beda_foil):
        # warna JAUH dari foil di sekitarnya, BUKAN warna meja, dan
        #   (tidak lebih gelap dari foil DAN kurang biru dari foil)
        #   ATAU jauh lebih terang dari foil (obat putih di bawah cahaya kebiruan)
        #   ATAU berwarna mencolok (mis. kapsul merah)
        if jarak_foil[i] <= beda_foil or jarak_meja[i] < BEDA_MEJA:
            return False
        if abu[i] >= RASIO_GELAP * foil_lokal[i] and biru[i] < biru_foil[i] - SELISIH_BIRU:
            return True
        if abu[i] >= RASIO_TERANG * acuan_terang[i]:
            return True                      # jauh lebih terang dari foil
        return sat[i] > BATAS_MENCOLOK and nilai_v[i] >= RASIO_GELAP * foil_lokal[i]

    # E1. BIJI (seed) = di dalam blister, HALUS, dan mirip obat
    biji = [0] * n
    boleh = [0] * n
    for i in range(n):
        if not blister[i]:
            continue
        if mirip_obat(i, BEDA_FOIL_TUMBUH):
            boleh[i] = 1
            if std[i] < BATAS_HALUS and mirip_obat(i, BEDA_FOIL):
                biji[i] = 1
    biji = opening(biji, w, h, 3)

    # E2. REGION GROWING: biji dilebarkan (dilasi 3x3) berulang kali, tetapi
    #     hanya ke piksel yang 'boleh' (mirip obat dengan syarat lebih longgar,
    #     tanpa syarat halus). Tepi obat & sambungan kapsul 2 warna ikut terisi.
    calon = biji
    for _ in range(LANGKAH_TUMBUH):
        tumbuh = dilasi(calon, w, h, 3)
        calon = [1 if (t and bo) or c else 0 for t, bo, c in zip(tumbuh, boleh, calon)]
    calon = isi_lubang(closing(calon, w, h, 5), w, h)
    calon = opening(calon, w, h, 5)
    _, daftar = label_komponen(calon, w, h)

    # putih jenuh (pantulan) diukur pada citra SEBELUM gamma/stretching
    kilau = [1 if (rb[i] >= BATAS_KILAU and gb[i] >= BATAS_KILAU and bb[i] >= BATAS_KILAU) else 0
             for i in range(n)]

    # ---------------- F. penilaian region -------------
    lapor("F. Menilai setiap region (deskriptor)...")
    def nilai_region(d):
        # menghitung deskriptor region lalu memutuskan: obat atau ditolak (alasan)
        cx, cy, isi, lonjong = momen(d["piksel"], w)
        d["cx"] = cx
        d["cy"] = cy
        d["isi"] = isi
        d["lonjong"] = lonjong
        d["warna"] = rata_warna(R, G, B, d["piksel"])
        d["d_meja"] = jarak3(d["warna"], warna_meja) if warna_meja else 999
        # kode rantai -> keliling -> kekompakan = 4*pi*luas / keliling^2
        # (informasi bentuk tambahan, 1 = lingkaran sempurna)
        d["kode_rantai"] = kode_rantai(d["piksel"], w)
        d["keliling"] = keliling_dari_rantai(d["kode_rantai"])
        d["kompak"] = 4 * 3.14159265 * d["luas"] / (d["keliling"] ** 2 + 1)
        ss = 0.0
        st = 0.0
        sfl = 0.0
        sj = 0.0
        sb = 0.0
        sbf = 0.0
        nk = 0
        nd = 0
        nb = 0
        for i in d["piksel"]:
            ss += sat[i]
            st += abu[i]
            sfl += acuan_terang[i]
            sj += jarak_foil[i]
            sb += biru[i]
            sbf += biru_foil[i]
            nk += kilau[i]
            nd += sekitar_blister[i]
            nb += biji[i]
        luas = d["luas"]
        d["sat"] = ss / luas
        d["terang"] = st / (sfl + 1)
        d["d_foil"] = sj / luas
        d["selisih_biru"] = (sbf - sb) / luas
        d["kilau"] = nk / luas
        d["dalam"] = nd / luas
        d["biji"] = nb / luas
        # CINCIN di sekeliling region: berapa bagian yang berwarna meja?
        # Obat di kantong dikelilingi plastik/foil, pantulan cahaya di meja
        # dikelilingi meja.
        d["cincin_meja"] = cincin_meja(d, jarak_meja, w, h)
        if d["dalam"] < MIN_DALAM:
            return "luar"
        if d["d_meja"] < BEDA_MEJA:
            return "kosong"
        # kromatisitas (r, g) = (R, G) / (R+G+B): tidak bergantung kecerahan,
        # jadi meja yang terlihat lewat plastik dan terkena flash tetap dikenali
        if warna_meja and jarak_kromatis(d["warna"], warna_meja) < BEDA_KROMATIS_MEJA:
            return "kosong"
        if d["cincin_meja"] > MAKS_CINCIN_MEJA:
            return "di meja"
        if d["biji"] < MIN_BIJI:
            return "kasar"
        if d["d_foil"] < BEDA_FOIL_REGION:
            return "foil"
        if d["selisih_biru"] < SELISIH_BIRU_REGION and d["sat"] < BATAS_MENCOLOK                 and d["terang"] < RASIO_TERANG_REGION:
            return "plastik"
        if d["kilau"] >= MAKS_KILAU:
            return "pantulan"
        if isi < MIN_ISI or lonjong > MAKS_LONJONG:
            return "bentuk"
        return ""

    def pecah_region(d, k):
        # OPENING (erosi lalu dilasi) dengan structuring element k x k pada
        # region ini saja -> "jembatan" tipis yang menyambung obat dengan
        # silau / plastik terputus. Hasilnya dilabel ulang.
        x0 = w
        y0 = h
        x1 = 0
        y1 = 0
        for i in d["piksel"]:
            x = i % w
            y = i // w
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
        lw = x1 - x0 + 1
        lh = y1 - y0 + 1
        m = [0] * (lw * lh)
        for i in d["piksel"]:
            m[(i // w - y0) * lw + (i % w - x0)] = 1
        _, bagian = label_komponen(opening(m, lw, lh, k), lw, lh)
        hasil = []
        for bg in bagian:
            bg["piksel"] = [(j // lw + y0) * w + (j % lw + x0) for j in bg["piksel"]]
            hasil.append(bg)
        return hasil

    lolos = []
    ditolak = []
    antrian = [(d, 0) for d in daftar]
    while antrian:
        d, tingkat = antrian.pop()
        if d["luas"] < MIN_LUAS_RELATIF * luas_blister:
            continue
        d["alasan"] = nilai_region(d)
        if d["alasan"] in ("bentuk", "luar") and tingkat < len(PECAH_BERTINGKAT):
            bagian = pecah_region(d, PECAH_BERTINGKAT[tingkat])
            if len(bagian) >= 2 or (len(bagian) == 1 and bagian[0]["luas"] < 0.9 * d["luas"]):
                for bg in bagian:
                    antrian.append((bg, tingkat + 1))
                continue
        (ditolak if d["alasan"] else lolos).append(d)

    obat = []
    if lolos:
        luas_urut = sorted_manual([d["luas"] for d in lolos])
        # patokan 1 butir = median luas dari separuh region terbesar
        besar = luas_urut[len(luas_urut) // 2:]
        median_luas = besar[len(besar) // 2]
        for d in lolos:
            rasio = d["luas"] / median_luas
            d["rasio"] = rasio
            if rasio < BATAS_SERPIHAN:
                d["alasan"] = "serpihan"
                ditolak.append(d)
            else:
                d["potongan"] = rasio < BATAS_POTONGAN
                obat.append(d)

    if catat is not None:
        catat("kotak %s  potong %dx%d  A:%s eta %.2f  blister: %s  meja %s  foil %s" %
              (kasar["kotak"], w, h, kasar["mode"], kasar["eta"], sumber_blister, warna_meja, warna_foil))
        for d in sorted(obat + ditolak, key=lambda d: -d["luas"]):
            catat("  luas %5d (%3d,%3d) isi %.2f lj %.2f rgb(%3d,%3d,%3d) dM %3.0f kM %.3f dF %3.0f dB %+.3f ter %.2f sat %.2f kil %.2f dlm %.2f cin %.2f biji %.2f -> %s" %
                  (d["luas"], d["cx"], d["cy"], d["isi"], d["lonjong"], d["warna"][0], d["warna"][1],
                   d["warna"][2], d["d_meja"], jarak_kromatis(d["warna"], warna_meja) if warna_meja else 9, d["d_foil"], d["selisih_biru"], d["terang"], d["sat"], d["kilau"], d["dalam"],
                   d["cincin_meja"], d["biji"], d["alasan"] or "OBAT"))

    hasil = {
        "lebar": w, "tinggi": h,
        "asli": potongan_asli,            # potongan blister (sebelum perbaikan)
        "diperbaiki": (R, G, B),          # setelah perbaikan citra (tahap C)
        "langkah": langkah,               # daftar perbaikan yang dipilih otomatis
        "skor_foil": skor, "foil": foil, "batas_foil": batas_foil,
        "cb": seg["cb"], "mode_blister": seg["mode"], "eta_cb": seg["eta"],
        "blister_asli": blister_asli, "blister": blister, "sumber_blister": sumber_blister,
        "std": std, "jarak_foil": jarak_foil, "biji": biji, "calon": calon, "kilau": kilau,
        "warna_meja": warna_meja, "warna_foil": warna_foil,
        "obat": obat, "ditolak": ditolak,
        "jumlah": len(obat), "jumlah_potongan": sum(1 for d in obat if d["potongan"]),
        "kasar": kasar, "waktu": time.time() - mulai,
    }
    return hasil
