# =============================================================================
# DETEKSI SISA OBAT DALAM KEMASAN BLISTER
# Digital Image Processing MURNI - tanpa Machine Learning / Computer Vision
# Semua rumus ditulis manual di rumus_dip.py
#
# Alur:
#   1. Baca foto, perkecil (sampling) -> cepat, tidak ngelag
#   2. ANALISIS KONDISI gambar -> program MEMILIH materi yang dipakai:
#        salt & pepper     -> Median filter 3x3            (03 Spatial)
#        noise periodik    -> DFT + Butterworth notch      (04 Frequency)
#        gaussian noise    -> Gaussian lowpass 5x5         (03 Spatial)
#        histogram sempit  -> Histogram equalization       (02 Enhancement)
#        kontras rendah    -> Contrast stretching          (02 Enhancement)
#        terlalu gelap     -> Gamma < 1                    (02 Enhancement)
#        terlalu terang    -> Gamma > 1                    (02 Enhancement)
#   3. Cari area blister (ROI): foil itu TERANG, TIDAK BERWARNA, dan KASAR
#        terang x (1 - saturasi) -> Otsu, Sobel -> kepadatan tepi -> Otsu
#        -> closing -> isi lubang -> opening (buang silau meja)
#      Lalu blister DIPOTONG dari foto asli (resolusi penuh) supaya obat
#      cukup besar, dan langkah 2 diulang pada potongan itu.
#   4. Calon obat (TIDAK bergantung warna obat): di dalam ROI, obat itu
#      HALUS (std lokal kecil), warnanya JAUH dari foil (lebih terang /
#      lebih gelap / berwarna) dan BEDA dengan meja. Setiap calon dinilai:
#        ukuran wajar, bentuk kompak (momen), dikelilingi blister (oktagon),
#        KONTRAS dengan cincin di sekelilingnya, TEPI TEGAS (Sobel).
#      Calon yang menempel ke foil datar dipecah dengan opening bertingkat.
#   5. Region growing dari setiap inti -> mask mengikuti BENTUK ASLI obat
#      (bulat, kapsul, lonjong), kontur digambar di atas foto.
#   6. Saring region dengan aturan:
#        bentuk (momen: isi elips, lonjong), ukuran mirip obat terbesar,
#        luas ~2x median = 2 butir menempel
#
# Cara pakai:
#   python deteksi_sisa_obat.py                -> jendela (pilih gambar)
#   python deteksi_sisa_obat.py gambar.jpg     -> hasil di terminal saja
# =============================================================================

import sys
import time
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rumus_dip import *


# ----------------------------- THRESHOLD -------------------------------------
UKURAN_KERJA     = 360    # sisi terpanjang foto utuh setelah diperkecil
UKURAN_OBAT      = 450    # sisi terpanjang potongan blister
MARGIN_POTONG    = 12     # px (gambar kecil) di sekeliling blister ikut dipotong
BATAS_SALT       = 0.01   # > 1% piksel ekstrem yg beda jauh dari median -> salt & pepper
BATAS_NOISE      = 6.0    # residu noise area datar > 6 -> gaussian noise
BATAS_PUNCAK     = 100    # puncak spektrum > 100x median spektrum -> noise periodik
RADIUS_PUSAT     = 10     # frekuensi rendah (isi gambar) tidak dianggap noise
BATAS_STD        = 40     # std keabuan < 40 -> kontras rendah
BATAS_RENTANG    = 70     # rentang p1..p99 < 70 -> histogram sempit -> equalization
BATAS_GELAP      = 70     # rata-rata < 70 -> terlalu gelap
BATAS_TERANG     = 185    # rata-rata > 185 -> terlalu terang
RADIUS_NOTCH     = 9      # D0 Butterworth notch reject

JENDELA_TEKSTUR  = 7      # ukuran filter mean untuk kepadatan tepi (ROI)
CLOSING_ROI      = 13     # closing menyatukan foil blister
OPENING_ROI      = 17     # opening besar memutus silau meja yg menempel ke blister
PITA_TEPI_FOTO   = 6      # pita di tepi foto diabaikan saat mencari ROI

JENDELA_STD      = 5      # jendela std lokal (kehalusan)
SKALA_STD        = 40     # std 40 -> 255 (std > 40 pasti kasar)
BEDA_FOIL        = 40     # jarak warna ke foil > 40 (lebih terang/gelap/berwarna)
MIN_TAJAM        = 60     # rata-rata Sobel di garis batas obat (tepi tegas)
BEDA_SEKITAR     = 30     # warna calon obat beda > 30 dari cincin di sekelilingnya
PISAH_BERTINGKAT = (9, 13, 17)  # opening bertingkat memutus obat <-> foil datar
LEBAR_CINCIN     = 6      # lebar cincin pembanding (px)
MIN_DIKELILINGI  = 0.85   # >= 85% cincin harus di dalam blister
MAKS_LUAS_ROI    = 0.15   # calon > 15% luas blister = foil datar, bukan obat
MAKS_CINCIN_MEJA = 0.40   # >= 40% cincin berwarna meja -> pantulan di kantong kosong
BEDA_MEJA        = 45     # jarak warna ke meja < 45 -> kantong kosong / meja
MIN_INTI         = 40     # region < 40 piksel diabaikan
LANGKAH_TUMBUH   = 8      # region growing maksimal 8 piksel ke luar
BATAS_TUMBUH     = 50     # piksel ikut tumbuh bila jarak warna ke obat < 50
MAKS_TUMBUH      = 2.0    # tumbuh > 2x inti = bocor ke foil -> batal
OPENING_PISAH    = 11     # opening 11x11 memutus sambungan tipis antar-region
MIN_ISI          = 0.75   # luas / luas elips momen (obat utuh ~0.95)
MAKS_LONJONG     = 4.0    # sumbu panjang / sumbu pendek
BATAS_KECIL      = 0.35   # luas < 0.35 x obat terbesar -> serpihan, bukan obat
BATAS_KECIL_MEDIAN = 0.5  # luas < 0.5 x median obat -> serpihan
MIN_LUAS_ROI     = 0.003  # obat minimal 0.3% luas blister (lebih kecil = serpihan/teks)
MIN_ISI_DOBEL    = 0.87   # "2 butir menempel" hanya bila bentuk tetap kompak
BATAS_KILAU      = 235    # piksel min(R,G,B) >= 235 = putih jenuh = kilau/pantulan
LEBAR_SILAU      = 7      # zona silau = dilasi 7x7 dari titik jenuh (+-3 px)
MAKS_KILAU       = 0.90   # region >= 90% di zona silau -> pantulan, bukan obat
                          # (obat putih terang paling tinggi ~74%)
BATAS_DOBEL      = 1.60   # luas >= 1.6 x median -> dihitung round(luas/median) butir


# ======================= LANGKAH 2: ANALISIS KONDISI =========================
def cek_salt_pepper(abu):
    median = filter_median3(abu)
    ekstrem = (abu <= 8) | (abu >= 247)
    beda = (abu - median) ** 2 > 50 * 50
    jumlah = (ekstrem & beda).sum()
    return jumlah / abu.size


def cek_gaussian(abu):
    # residu = piksel - rata-rata 4 tetangga (M8). Noise diambil dari
    # blok 8x8 yang paling datar (persentil 25%) agar tepi tidak ikut.
    tinggi = abu.shape[0]
    lebar = abu.shape[1]
    p = pad_tepi(abu, 1)
    tetangga = (p[0:tinggi, 1:lebar + 1] + p[2:tinggi + 2, 1:lebar + 1] +
                p[1:tinggi + 1, 0:lebar] + p[1:tinggi + 1, 2:lebar + 2]) / 4
    residu = abu - tetangga
    residu[residu < 0] = -residu[residu < 0]

    nilai_blok = []
    for by in range(tinggi // 8):
        for bx in range(lebar // 8):
            blok = residu[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8]
            nilai_blok.append(blok.sum() / 64)
    nilai_blok.sort()
    return nilai_blok[len(nilai_blok) // 4]


def cek_periodik(abu):
    # DFT 2D -> cari puncak terang di luar pusat spektrum
    tinggi = abu.shape[0]
    lebar = abu.shape[1]
    F = geser_tengah(dft2(abu))
    besar = (F.real ** 2 + F.imag ** 2) ** 0.5

    # pembanding = median besar spektrum (nilai "biasa" sebuah frekuensi)
    cy = tinggi // 2
    cx = lebar // 2
    pusat = np.zeros((tinggi, lebar))
    for y in range(tinggi):
        for x in range(lebar):
            if (y - cy) ** 2 + (x - cx) ** 2 < RADIUS_PUSAT ** 2:
                pusat[y, x] = 1
    median = nilai_tengah(besar[pusat < 0.5].tolist())
    rasio = besar / (median + 1)
    rasio[pusat > 0.5] = 0                # frekuensi rendah = isi gambar, bukan noise
    rasio[:cy, :] = 0                     # cukup separuh (spektrum simetris)

    puncak = []
    tertinggi = 0
    for k in range(4):                    # maksimal 4 pasang puncak
        terbaik = 0
        py = 0
        px = 0
        baris = rasio.tolist()
        for y in range(cy, tinggi):
            for x in range(lebar):
                if baris[y][x] > terbaik:
                    terbaik = baris[y][x]
                    py = y
                    px = x
        if k == 0:
            tertinggi = terbaik
        if terbaik < BATAS_PUNCAK:
            break
        puncak.append((py, px))
        rasio[py - 4:py + 5, px - 4:px + 5] = 0
    return F, besar, puncak, tertinggi


def perbaiki_gambar(img):
    langkah = []        # (materi, nilai, alasan, dipakai?)
    abu = ke_abu(img)

    # --- 1. salt & pepper -> median ---
    salt = cek_salt_pepper(abu)
    if salt > BATAS_SALT:
        img = per_kanal(img, filter_median3)
        langkah.append(["Median filter 3x3", round(salt * 100, 2), "piksel ekstrem " +
                        str(round(salt * 100, 2)) + "% > " + str(BATAS_SALT * 100) + "%", True])
    else:
        langkah.append(["Median filter 3x3", round(salt * 100, 2), "salt&pepper " +
                        str(round(salt * 100, 2)) + "% (kecil)", False])
    abu = ke_abu(img)

    # --- 2. noise periodik -> DFT + Butterworth notch reject ---
    F, besar, puncak, tertinggi = cek_periodik(abu)
    spektrum = besar
    if len(puncak) > 0:
        H = butterworth_notch_reject(abu.shape[0], abu.shape[1], puncak, RADIUS_NOTCH, 2)
        for c in range(3):
            Fc = geser_tengah(dft2(img[:, :, c]))
            img[:, :, c] = idft2(geser_tengah(Fc * H)).real
        img = potong_nilai(img, 0, 255)
        langkah.append(["DFT + Butterworth notch", round(tertinggi), str(len(puncak)) +
                        " puncak, rasio " + str(round(tertinggi)) + " > " + str(BATAS_PUNCAK), True])
    else:
        langkah.append(["DFT + Butterworth notch", round(tertinggi), "puncak spektrum " +
                        str(round(tertinggi)) + " (tidak ada)", False])
    abu = ke_abu(img)

    # --- 3. gaussian noise -> gaussian lowpass ---
    noise = cek_gaussian(abu)
    if noise > BATAS_NOISE:
        img = per_kanal(img, konvolusi, kernel_gaussian(5, 1.2))
        langkah.append(["Gaussian lowpass 5x5", round(noise, 2), "noise " + str(round(noise, 2)) +
                        " > " + str(BATAS_NOISE), True])
    else:
        langkah.append(["Gaussian lowpass 5x5", round(noise, 2), "noise " + str(round(noise, 2)) +
                        " (kecil)", False])
    abu = ke_abu(img)

    bersih = img.copy()     # sudah bebas noise, intensitas masih asli (dipakai ROI)

    # --- 4. intensitas: equalization / stretching / gamma ---
    hist_awal = histogram(abu)
    rata, std = rata_dan_std(hist_awal)
    p1 = persentil(hist_awal, 0.01)
    p99 = persentil(hist_awal, 0.99)
    rentang = p99 - p1

    if rentang < BATAS_RENTANG:
        img = per_kanal(img, pakai_lut, lut_equalization(hist_awal))
        langkah.append(["Histogram equalization", rentang, "rentang p1-p99 = " + str(rentang) +
                        " < " + str(BATAS_RENTANG), True])
    else:
        langkah.append(["Histogram equalization", rentang, "rentang " + str(rentang) + " cukup", False])

    if rentang >= BATAS_RENTANG and std < BATAS_STD:
        img = per_kanal(img, pakai_lut, lut_stretching(p1, p99))
        langkah.append(["Contrast stretching", round(std, 1), "std " + str(round(std, 1)) +
                        " < " + str(BATAS_STD) + " -> tarik [" + str(p1) + ".." + str(p99) + "]", True])
    else:
        langkah.append(["Contrast stretching", round(std, 1), "std " + str(round(std, 1)), False])

    hist_baru = histogram(ke_abu(img))
    rata_baru, std_baru = rata_dan_std(hist_baru)
    if rata_baru < BATAS_GELAP:
        img = per_kanal(img, pakai_lut, lut_gamma(0.5))
        langkah.append(["Gamma 0.5 (terangkan)", round(rata_baru), "rata-rata " + str(round(rata_baru)) +
                        " < " + str(BATAS_GELAP), True])
    elif rata_baru > BATAS_TERANG:
        img = per_kanal(img, pakai_lut, lut_gamma(2.0))
        langkah.append(["Gamma 2.0 (gelapkan)", round(rata_baru), "rata-rata " + str(round(rata_baru)) +
                        " > " + str(BATAS_TERANG), True])
    else:
        langkah.append(["Gamma correction", round(rata_baru), "rata-rata " + str(round(rata_baru)) +
                        " normal", False])

    hasil = {"img": img, "bersih": bersih, "langkah": langkah, "spektrum": spektrum, "puncak": puncak,
             "hist_awal": hist_awal, "hist_akhir": histogram(ke_abu(img))}
    return hasil


# ===================== LANGKAH 3: CARI BLISTER (ROI) ========================
def nilai_tengah(daftar):
    urut = sorted(daftar)
    return urut[len(urut) // 2]


def bungkus_oktagon(mask):
    # Blister berbentuk persegi (boleh miring) -> tidak punya coakan.
    # Mask dibungkus oktagon: batas min/maks dari x, y, x+y, x-y
    # (x+y dan x-y = arah diagonal, sama seperti mencari 4 sudut blister).
    # Coakan di pinggir karena bayangan / obat di tepi ikut tertutup.
    tinggi = mask.shape[0]
    lebar = mask.shape[1]
    yy = np.zeros((tinggi, lebar))
    xx = np.zeros((tinggi, lebar))
    for y in range(tinggi):
        yy[y, :] = y
    for x in range(lebar):
        xx[:, x] = x
    ada = mask > 0.5
    hasil = np.ones((tinggi, lebar))
    for nilai in (xx, yy, xx + yy, xx - yy):
        v = nilai[ada]
        hasil = hasil * (nilai >= v.min()) * (nilai <= v.max())
    return hasil


def cari_blister(img):
    # Foil blister = TERANG + TIDAK BERWARNA (abu-abu mengkilap) + KASAR (berpola).
    #   meja kayu  -> berwarna (saturasi tinggi)  -> skor kecil
    #   meja gelap -> gelap                       -> skor kecil
    #   silau/pantulan di meja -> terang tapi HALUS -> tersaring tekstur
    # Warna OBAT tidak dipakai: lubang di mask diisi (isi_lubang).
    abu = ke_abu(img)
    tinggi = abu.shape[0]
    lebar = abu.shape[1]

    skor = konvolusi(abu * (1 - saturasi(img)), kernel_mean(JENDELA_TEKSTUR))
    skor_8bit = potong_nilai(skor * 255 / (skor.max() + 0.001), 0, 255)
    batas_skor = otsu(histogram(skor_8bit))

    tepi = sobel(abu)
    tekstur = konvolusi(tepi, kernel_mean(JENDELA_TEKSTUR))
    tekstur_8bit = tekstur * 255 / (tekstur.max() + 0.001)
    batas_tekstur = otsu(histogram(tekstur_8bit))

    foil = (skor_8bit > batas_skor) * (tekstur_8bit > batas_tekstur) * 1.0

    # pita tipis di tepi foto diabaikan (efek batas filter / DFT)
    p = PITA_TEPI_FOTO
    foil[:p, :] = 0
    foil[tinggi - p:, :] = 0
    foil[:, :p] = 0
    foil[:, lebar - p:] = 0

    roi = closing(foil, CLOSING_ROI)
    roi = isi_lubang(roi)
    roi = opening(roi, OPENING_ROI)          # putus "ekor" silau yg menempel
    roi = region_terbesar(roi)
    if roi.sum() < 0.01 * abu.size:          # blister tidak ketemu -> seluruh foto
        roi = np.ones(abu.shape)
    return {"tekstur": tekstur_8bit, "skor": skor_8bit, "roi": roi,
            "batas_roi": batas_tekstur, "batas_skor": batas_skor}


def potong_blister(asli, kecil, roi):
    # Kotak ROI di gambar kecil -> potong dari foto ASLI (resolusi penuh),
    # lalu perkecil lagi. Obat jadi jauh lebih besar dibanding diperkecil
    # langsung dari seluruh foto (foto HP: obat ~15 px -> ~50 px).
    f = asli.shape[0] / kecil.shape[0]
    y0 = kecil.shape[0]
    y1 = 0
    x0 = kecil.shape[1]
    x1 = 0
    for y in range(roi.shape[0]):
        baris = roi[y]
        if baris.max() > 0.5:
            if y < y0:
                y0 = y
            y1 = y
            for x in range(roi.shape[1]):
                if baris[x] > 0.5:
                    if x < x0:
                        x0 = x
                    if x > x1:
                        x1 = x
    m = MARGIN_POTONG
    ay0 = max(int((y0 - m) * f), 0)
    ay1 = min(int((y1 + m + 1) * f), asli.shape[0])
    ax0 = max(int((x0 - m) * f), 0)
    ax1 = min(int((x1 + m + 1) * f), asli.shape[1])

    sisi = max(ay1 - ay0, ax1 - ax0)
    langkah = (sisi + UKURAN_OBAT - 1) // UKURAN_OBAT
    if langkah < 1:
        langkah = 1
    potong = asli[ay0:ay1:langkah, ax0:ax1:langkah].copy()
    potong = potong[:potong.shape[0] // 2 * 2, :potong.shape[1] // 2 * 2]

    # ROI ikut dipindah ke koordinat potongan (piksel terdekat)
    roi_potong = np.zeros((potong.shape[0], potong.shape[1]))
    for y in range(potong.shape[0]):
        ky = int((ay0 + y * langkah) / f)
        if ky > kecil.shape[0] - 1:
            ky = kecil.shape[0] - 1
        for x in range(potong.shape[1]):
            kx = int((ax0 + x * langkah) / f)
            if kx > kecil.shape[1] - 1:
                kx = kecil.shape[1] - 1
            roi_potong[y, x] = roi[ky, kx]
    kotak = (x0, y0, x1, y1)
    return potong, roi_potong, kotak


# ===================== LANGKAH 4-6: CARI OBAT ===============================
def tumbuhkan_region(img, label, d, jarak_foil, jarak_meja, roi, terpakai):
    # REGION GROWING: mulai dari inti obat (seed), lalu berulang kali
    # dilasi 3x3 tetapi HANYA ke piksel yang warnanya lebih mirip obat
    # daripada foil / meja. Hasilnya mengikuti bentuk asli obat.
    tinggi = img.shape[0]
    lebar = img.shape[1]
    r = LANGKAH_TUMBUH
    y0 = max(d["y0"] - r, 0)
    y1 = min(d["y1"] + r + 1, tinggi)
    x0 = max(d["x0"] - r, 0)
    x1 = min(d["x1"] + r + 1, lebar)

    warna = warna_rata(img, d["piksel"])
    jarak_obat = jarak_ke_warna(img[y0:y1, x0:x1], warna)
    boleh = (jarak_obat < jarak_foil[y0:y1, x0:x1]) * (jarak_obat < BATAS_TUMBUH)
    boleh = boleh * roi[y0:y1, x0:x1] * (1 - terpakai[y0:y1, x0:x1])
    if jarak_meja is not None:
        boleh = boleh * (jarak_obat < jarak_meja[y0:y1, x0:x1])

    inti = (label[y0:y1, x0:x1] == d["nomor"]) * 1.0
    tumbuh = inti.copy()
    for i in range(r):
        baru = filter_maksimum(tumbuh, 3) * boleh
        baru[inti > 0.5] = 1
        tumbuh = baru
    tumbuh = isi_lubang(opening(closing(tumbuh, 5), 3))
    tumbuh = region_terbesar(tumbuh)
    if tumbuh.sum() > MAKS_TUMBUH * d["luas"] or tumbuh.sum() == 0:
        tumbuh = inti                        # bocor ke foil -> pakai inti saja
    else:
        _, daftar = label_region(tumbuh)
        _, _, isi, lonjong = momen_region(daftar[0]["piksel"])
        if isi < MIN_ISI:
            tumbuh = inti                    # bentuk rusak (bocor) -> inti saja

    hasil = np.zeros((tinggi, lebar))
    hasil[y0:y1, x0:x1] = tumbuh
    return hasil


def pisahkan_sambungan(d, ukuran):
    # Obat kadang tersambung ke pantulan / area cetakan foil lewat "jembatan"
    # tipis. Opening besar (erosi lalu dilasi) memutus jembatan itu.
    # Bila region pecah jadi >= 2 bagian, setiap bagian dinilai sendiri.
    # Bila tidak pecah, region asli dipakai (bentuknya tetap utuh).
    y0 = max(d["y0"] - 1, 0)
    x0 = max(d["x0"] - 1, 0)
    y1 = min(d["y1"] + 2, ukuran[0])
    x1 = min(d["x1"] + 2, ukuran[1])
    m = np.zeros((y1 - y0, x1 - x0))
    for (y, x) in d["piksel"]:
        m[y - y0, x - x0] = 1
    label, bagian = label_region(opening(m, OPENING_PISAH))
    hasil = []
    for b in bagian:
        if b["luas"] < MIN_INTI:
            continue
        b["piksel"] = [(y + y0, x + x0) for (y, x) in b["piksel"]]
        b["y0"] += y0
        b["y1"] += y0
        b["x0"] += x0
        b["x1"] += x0
        hasil.append(b)
    if len(hasil) >= 2:
        return hasil
    return [d]


def keliling_cincin(d, r):
    # luas cincin bila region TIDAK terpotong tepi gambar
    m = np.zeros((d["y1"] - d["y0"] + 1 + 2 * r + 2, d["x1"] - d["x0"] + 1 + 2 * r + 2))
    for (y, x) in d["piksel"]:
        m[y - d["y0"] + r + 1, x - d["x0"] + r + 1] = 1
    return (filter_maksimum(m, 2 * r + 1) - filter_maksimum(m, 5)).sum()


def nilai_calon(img, roi, d, luas_roi, jarak_meja=None):
    # Menilai satu calon obat TANPA melihat warna tertentu:
    #   1. ukuran wajar dibanding blister
    #   2. bentuk kompak (momen: isi elips, lonjong)
    #   3. dikelilingi blister (cincin di sekitarnya hampir semua di ROI)
    #   4. KONTRAS dengan cincin sekitarnya (obat putih, merah, kuning,
    #      cokelat, ... semuanya beda dari foil/kantong di sekelilingnya;
    #      foil datar & pantulan warnanya sama dengan sekelilingnya)
    #   5. tepi tegas (Sobel di garis batas)
    #   6. bukan pantulan di KANTONG KOSONG: kantong kosong itu transparan,
    #      jadi di sekeliling kilau plastiknya terlihat warna MEJA. Obat
    #      menutupi kantongnya, sehingga meja tidak terlihat di sekitarnya.
    # hasil: "ok" atau alasan ditolak
    if d["luas"] < MIN_INTI:
        return "kecil"
    if d["luas"] > MAKS_LUAS_ROI * luas_roi:
        return "besar"
    cy, cx, isi, lonjong = momen_region(d["piksel"])
    if isi < MIN_ISI or lonjong > MAKS_LONJONG:
        return "bentuk"

    tinggi = img.shape[0]
    lebar = img.shape[1]
    r = LEBAR_CINCIN + 2
    y0 = max(d["y0"] - r, 0)
    y1 = min(d["y1"] + r + 1, tinggi)
    x0 = max(d["x0"] - r, 0)
    x1 = min(d["x1"] + r + 1, lebar)
    m = np.zeros((y1 - y0, x1 - x0))
    for (y, x) in d["piksel"]:
        m[y - y0, x - x0] = 1
    cincin = (filter_maksimum(m, 2 * r + 1) - filter_maksimum(m, 5)) * roi[y0:y1, x0:x1]
    if cincin.sum() < MIN_DIKELILINGI * keliling_cincin(d, r):
        return "tepi"
    potong = img[y0:y1, x0:x1]
    dalam = median_warna(potong, m)
    luar = median_warna(potong, cincin)
    beda = ((dalam[0] - luar[0]) ** 2 + (dalam[1] - luar[1]) ** 2 + (dalam[2] - luar[2]) ** 2) ** 0.5
    if beda < BEDA_SEKITAR:
        return "kontras"

    # 5. TEPI TEGAS: rata-rata Sobel di pita garis batas region.
    #    Obat punya batas yang tajam; noda/bayangan di foil memudar pelan.
    pita = filter_maksimum(m, 7) - filter_minimum(m, 5)
    tepi = sobel(ke_abu(potong))
    if (tepi * pita).sum() / (pita.sum() + 1) < MIN_TAJAM:
        return "pudar"

    if jarak_meja is not None:
        mirip_meja = (jarak_meja[y0:y1, x0:x1] < BEDA_MEJA) * cincin
        if mirip_meja.sum() >= MAKS_CINCIN_MEJA * cincin.sum():
            return "pantulan"
    return "ok"


def geser_region(b, y0, x0):
    b["piksel"] = [(y + y0, x + x0) for (y, x) in b["piksel"]]
    b["y0"] += y0
    b["y1"] += y0
    b["x0"] += x0
    b["x1"] += x0
    return b


def pecah_dan_nilai(img, roi, d, luas_roi, tingkat, jarak_meja=None):
    # Nilai region d. Bila gagal karena BENTUK / terlalu BESAR (obat menempel
    # ke foil datar lewat jembatan), pecah dengan opening yang lebih besar
    # (PISAH_BERTINGKAT[tingkat]) lalu setiap bagian dinilai lagi (rekursif).
    alasan = nilai_calon(img, roi, d, luas_roi, jarak_meja)
    if alasan == "ok":
        return [d]
    if (alasan != "bentuk" and alasan != "besar") or tingkat >= len(PISAH_BERTINGKAT):
        return []
    y0 = d["y0"]
    x0 = d["x0"]
    m = np.zeros((d["y1"] - y0 + 1, d["x1"] - x0 + 1))
    for (y, x) in d["piksel"]:
        m[y - y0, x - x0] = 1
    _, bagian = label_region(opening(m, PISAH_BERTINGKAT[tingkat]))
    lolos = []
    for b in bagian:
        lolos = lolos + pecah_dan_nilai(img, roi, geser_region(b, y0, x0), luas_roi, tingkat + 1,
                                        jarak_meja)
    return lolos


def saring_calon(img, calon, roi, jarak_meja=None):
    tinggi = img.shape[0]
    lebar = img.shape[1]
    luas_roi = roi.sum()
    # coakan di tepi ROI (bayangan / obat di pinggir blister) ditutup dulu
    # dengan oktagon pembungkus supaya obat di pinggir tetap "dikelilingi"
    roi = bungkus_oktagon(roi)
    hasil = np.zeros((tinggi, lebar))
    label, daftar = label_region(calon)
    for d in daftar:
        if d["luas"] < MIN_INTI:
            continue
        for b in pecah_dan_nilai(img, roi, d, luas_roi, 0, jarak_meja):
            for (y, x) in b["piksel"]:
                hasil[y, x] = 1
    return hasil


def cari_obat(img, roi, kilau=None):
    abu = ke_abu(img)

    # --- 4a. kehalusan: std lokal -> Otsu (halus = calon obat, kasar = foil) ---
    std = std_lokal(abu, JENDELA_STD)
    std_8bit = potong_nilai(std * 255 / SKALA_STD, 0, 255)
    batas_std = otsu(histogram(std_8bit))
    halus = (std_8bit < batas_std) * roi
    kasar = (std_8bit >= batas_std) * roi

    # --- 4b. warna pembanding: foil (daerah kasar) dan meja (luar ROI) ---
    warna_foil = median_warna(img, kasar)
    jarak_foil = jarak_ke_warna(img, warna_foil)
    luar = 1 - filter_maksimum(roi, 9)
    warna_meja = None
    jarak_meja = None
    if luar.sum() > 0.03 * abu.size:
        warna_meja = median_warna(img, luar)
        jarak_meja = jarak_ke_warna(img, warna_meja)

    # --- 4c. CALON obat: halus + warnanya JAUH dari warna foil ---
    #         (ke arah mana pun: lebih terang, lebih gelap, atau berwarna ->
    #          obat putih, kuning, merah, cokelat, hitam ... semua bisa)
    #         + bukan warna meja (kantong kosong menampakkan meja)
    #         + bukan garis tepi kuat (Sobel) -> obat terpisah dari pantulan
    inti = halus * (jarak_foil > BEDA_FOIL)
    if jarak_meja is not None:
        inti = inti * (jarak_meja > BEDA_MEJA)
    tepi = sobel(abu)
    tepi_skala = tepi * 255 / (tepi.max() + 0.001)
    p99 = persentil(histogram(tepi_skala), 0.99)          # 1% tepi terkuat = 255
    tepi_8bit = potong_nilai(tepi_skala * 255 / (p99 + 1), 0, 255)
    hist_tepi = [0] * 256
    for v in tepi_8bit[roi > 0.5].astype(int).tolist():
        hist_tepi[v] += 1
    batas_tepi = otsu(hist_tepi)
    inti = inti * (tepi_8bit < batas_tepi)
    inti = opening(inti, 5)
    # --- 4d. setiap calon dinilai: ukuran, bentuk, dikelilingi blister,
    #         kontras dgn sekelilingnya, tepi tegas (lihat nilai_calon) ---
    inti = saring_calon(img, inti, roi, jarak_meja)
    inti = opening(isi_lubang(closing(inti, 9)), 5)

    # --- 5. region growing dari setiap inti -> bentuk obat utuh ---
    label, daftar_inti = label_region(inti)
    obat_mask = np.zeros(abu.shape)
    for d in daftar_inti:
        if d["luas"] < MIN_INTI:
            continue
        obat_mask = obat_mask + tumbuhkan_region(img, label, d, jarak_foil, jarak_meja, roi, obat_mask)
        obat_mask[obat_mask > 1] = 1
    # dua warna kapsul (merah-putih) tumbuh terpisah -> disatukan lagi
    obat_mask = isi_lubang(closing(obat_mask, 5))

    # --- 6. saring region dengan aturan ---
    label_akhir, daftar_awal = label_region(obat_mask)
    daftar = []
    for d in daftar_awal:
        if d["luas"] < MIN_INTI:
            continue
        daftar = daftar + pisahkan_sambungan(d, abu.shape)

    calon = []
    ditolak = []
    luas_roi = roi.sum()
    for d in daftar:
        d["cy"], d["cx"], d["isi"], d["lonjong"] = momen_region(d["piksel"])
        n_kilau = 0
        if kilau is not None:
            for (y, x) in d["piksel"]:
                n_kilau += kilau[y, x]
        d["kilau"] = n_kilau / d["luas"]
        if d["kilau"] >= MAKS_KILAU:
            d["alasan"] = "pantulan"
            ditolak.append(d)
        elif d["luas"] < MIN_LUAS_ROI * luas_roi:
            d["alasan"] = "serpihan"
            ditolak.append(d)
        elif d["isi"] < MIN_ISI:
            d["alasan"] = "bentuk"
            ditolak.append(d)
        elif d["lonjong"] > MAKS_LONJONG:
            d["alasan"] = "lonjong"
            ditolak.append(d)
        else:
            calon.append(d)

    # aturan ukuran: obat dalam satu blister sama besar.
    obat = []
    jumlah = 0
    if len(calon) > 0:
        # patokan = region berbentuk baik yang PALING BESAR
        # (serpihan plastik / pantulan kecil jadi tersaring)
        terbesar = 0
        for d in calon:
            if d["luas"] > terbesar:
                terbesar = d["luas"]
        sisa = []
        for d in calon:
            if d["luas"] < BATAS_KECIL * terbesar:
                d["alasan"] = "kecil"
                ditolak.append(d)
            else:
                sisa.append(d)

        # 1 butir = median luas yang lolos. Serpihan jauh di bawah median dibuang.
        if len(sisa) > 0:
            median_luas = nilai_tengah([d["luas"] for d in sisa])
            for d in sisa:
                if d["luas"] < BATAS_KECIL_MEDIAN * median_luas:
                    d["alasan"] = "kecil"
                    ditolak.append(d)
                    continue
                # ~2x median = 2 butir menempel, TAPI hanya bila bentuknya
                # tetap kompak (2 obat berdempet: isi elips ~0.9). Obat yang
                # bocor ke foil datar bentuknya tidak rapi -> tetap 1 butir.
                d["butir"] = 1
                if d["luas"] >= BATAS_DOBEL * median_luas and d["isi"] >= MIN_ISI_DOBEL:
                    d["butir"] = round(d["luas"] / median_luas)
                jumlah += d["butir"]
                obat.append(d)

    # mask per region (untuk menggambar kontur)
    for d in obat + ditolak:
        m = np.zeros(abu.shape)
        for (y, x) in d["piksel"]:
            m[y, x] = 1
        d["mask"] = m

    hasil = {"std": std_8bit, "inti": inti, "obat_mask": obat_mask,
             "obat": obat, "ditolak": ditolak, "jumlah": jumlah,
             "batas_std": batas_std, "batas_tepi": batas_tepi,
             "warna_foil": warna_foil, "warna_meja": warna_meja}
    return hasil


def warna_rata(img, piksel):
    total = [0, 0, 0]
    for (y, x) in piksel:
        for c in range(3):
            total[c] += img[y, x, c]
    return [total[0] / len(piksel), total[1] / len(piksel), total[2] / len(piksel)]


def analisis(path):
    mulai = time.time()
    asli = baca_gambar(path)
    kecil = perkecil(asli, UKURAN_KERJA)
    kecil = kecil[:kecil.shape[0] // 2 * 2, :kecil.shape[1] // 2 * 2]

    # tahap 1: seluruh foto (kecil) -> perbaikan -> cari blister
    perbaikan_kecil = perbaiki_gambar(kecil.copy())
    # ROI dicari pada gambar yg sudah bebas noise tapi BELUM diubah
    # intensitasnya: equalization/gamma membuat meja gelap ikut terang
    blister = cari_blister(perbaikan_kecil["bersih"])

    # tahap 2: potong blister dari foto asli -> perbaikan lagi -> cari obat
    potong, roi_potong, kotak = potong_blister(asli, kecil, blister["roi"])
    perbaikan = perbaiki_gambar(potong.copy())

    # PANTULAN CAHAYA (kilau): putih jenuh di ketiga kanal. Diukur pada
    # gambar SEBELUM stretching/gamma (supaya obat putih tidak ikut jenuh).
    bersih = perbaikan["bersih"]
    kilau = ((bersih[:, :, 0] >= BATAS_KILAU) * (bersih[:, :, 1] >= BATAS_KILAU) *
             (bersih[:, :, 2] >= BATAS_KILAU)) * 1.0
    # zona silau = titik jenuh + pinggiran 3 px (cahaya pantulan memudar
    # di sekeliling titik jenuh, pinggiran itu juga bagian pantulan)
    zona_silau = filter_maksimum(kilau, LEBAR_SILAU)
    deteksi = cari_obat(perbaikan["img"], roi_potong, zona_silau)
    deteksi["kilau"] = kilau

    d = {"nama": path.replace("\\", "/").split("/")[-1], "awal": kecil, "potong": potong,
         "roi_kecil": blister["roi"], "tekstur": blister["tekstur"], "kotak": kotak,
         "batas_roi": blister["batas_roi"], "batas_skor": blister["batas_skor"], "roi": roi_potong}
    d.update(perbaikan)
    d.update(deteksi)
    d["waktu"] = time.time() - mulai
    return d


# ============================== TAMPILAN =====================================
WARNA_OBAT = [0.0, 1.0, 0.2]      # hijau terang
WARNA_TOLAK = [1.0, 0.15, 0.15]   # merah
WARNA_KILAU = [1.0, 0.85, 0.0]    # kuning = pantulan cahaya


def gambar_overlay(img, obat, ditolak, kilau=None):
    # gambar + isi obat diwarnai transparan + kontur tebal mengikuti bentuk obat
    # pantulan cahaya (kilau) diberi warna KUNING transparan
    hasil = potong_nilai(img, 0, 255) / 255
    if kilau is not None:
        k = kilau > 0.5
        for c in range(3):
            kanal = hasil[:, :, c]
            kanal[k] = 0.5 * kanal[k] + 0.5 * WARNA_KILAU[c]
    for d in ditolak:
        garis = kontur(d["mask"])
        for c in range(3):
            kanal = hasil[:, :, c]
            kanal[garis > 0.5] = WARNA_TOLAK[c]
    for d in obat:
        isi = d["mask"] > 0.5
        garis = filter_maksimum(kontur(d["mask"]), 3) > 0.5        # kontur ditebalkan
        for c in range(3):
            kanal = hasil[:, :, c]
            kanal[isi] = 0.6 * kanal[isi] + 0.4 * WARNA_OBAT[c]
            kanal[garis] = WARNA_OBAT[c]
    return hasil


def figure_hasil(d):
    fig = Figure(figsize=(16, 7), dpi=80)

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(potong_nilai(d["awal"], 0, 255) / 255)
    x0, y0, x1, y1 = d["kotak"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", linewidth=2))
    ax.set_title("Foto input  (kuning = blister yang ditemukan)")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(gambar_overlay(d["img"], d["obat"], d["ditolak"], d["kilau"]))
    nomor = 0
    for o in d["obat"]:
        nomor += 1
        teks = str(nomor)
        if o["butir"] > 1:
            teks = str(nomor) + "-" + str(nomor + o["butir"] - 1)
            nomor += o["butir"] - 1
        ax.text(o["cx"], o["cy"], teks, color="white", fontsize=14, fontweight="bold",
                ha="center", va="center",
                bbox={"facecolor": "green", "alpha": 0.85, "boxstyle": "circle", "edgecolor": "none"})
    for o in d["ditolak"]:
        ax.text(o["cx"], o["cy"], o["alasan"], color="red", fontsize=8, ha="center", va="center")
    ax.set_title("Hijau = sisa obat, merah = ditolak, kuning = pantulan cahaya")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.axis("off")
    teks = "MATERI YANG DIPILIH OTOMATIS\n(pada potongan blister)\n\n"
    for l in d["langkah"]:
        if l[3]:
            tanda = "[DIPAKAI] "
        else:
            tanda = "[ - ]     "
        teks += tanda + l[0] + "\n           " + l[2] + "\n"
    teks += "\n[DIPAKAI] ROI: terang x (1-saturasi) > " + str(d["batas_skor"])
    teks += "\n          & Sobel tekstur > " + str(d["batas_roi"])
    teks += "\n[DIPAKAI] Potong blister dari foto asli"
    teks += "\n[DIPAKAI] Std lokal " + str(JENDELA_STD) + "x" + str(JENDELA_STD) + " + Otsu = " + str(d["batas_std"])
    teks += "\n[DIPAKAI] Calon: halus, beda dari foil & meja"
    teks += "\n          (warna obat APA PUN)"
    teks += "\n[DIPAKAI] Nilai calon: bentuk, kontras cincin,"
    teks += "\n          tepi tegas, dikelilingi blister"
    teks += "\n[DIPAKAI] Region growing (bentuk obat)"
    teks += "\n[DIPAKAI] Momen region (isi, lonjong)"
    teks += "\n[DIPAKAI] Pantulan: R,G,B >= " + str(BATAS_KILAU) + " (kuning),"
    teks += "\n          region >= " + str(int(MAKS_KILAU * 100)) + "% di zona silau ditolak"
    if d["warna_meja"] is None:
        teks += "\n\n! Meja hampir tidak terlihat:\n  cek warna meja dilewati"
    teks += "\n\nWaktu proses: " + str(round(d["waktu"], 2)) + " detik"
    ax.text(0, 1, teks, va="top", fontsize=10, family="monospace")

    fig.suptitle(d["nama"] + "   ==>   SISA OBAT: " + str(d["jumlah"]),
                 fontsize=18, fontweight="bold", color="darkgreen")
    fig.tight_layout()
    return fig


def figure_langkah(d):
    fig = Figure(figsize=(16, 9), dpi=80)

    ax = fig.add_subplot(2, 4, 1)
    ax.bar(range(256), d["hist_awal"], color="gray", width=1)
    ax.set_title("Histogram potongan (awal)")

    ax = fig.add_subplot(2, 4, 2)
    ax.bar(range(256), d["hist_akhir"], color="steelblue", width=1)
    ax.set_title("Histogram setelah perbaikan")

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(np.log(1 + d["spektrum"]), cmap="gray")
    for (py, px) in d["puncak"]:
        ax.add_patch(Rectangle((px - 6, py - 6), 12, 12, fill=False, edgecolor="red"))
    ax.set_title("Spektrum DFT (merah = noise)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(d["roi_kecil"], cmap="gray")
    ax.set_title("ROI blister (terang, abu-abu, kasar)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(potong_nilai(d["img"], 0, 255) / 255)
    ax.set_title("Potongan blister + perbaikan")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 6)
    ax.imshow(d["std"], cmap="magma")
    ax.set_title("Std lokal (gelap = halus)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(d["inti"], cmap="gray")
    ax.set_title("Inti obat (seed)")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 8)
    ax.imshow(d["obat_mask"], cmap="gray")
    ax.set_title("Setelah region growing")
    ax.axis("off")

    fig.suptitle("Langkah-langkah pengolahan", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def jalankan_jendela():
    root = tk.Tk()
    root.title("Deteksi Sisa Obat Blister - Digital Image Processing")
    root.geometry("1340x850")
    path_gambar = [""]

    atas = tk.Frame(root, padx=10, pady=8)
    atas.pack(side="top", fill="x")
    label_file = tk.Label(atas, text="Gambar: (belum dipilih)", width=60, anchor="w")
    status = tk.Label(atas, text="Pilih gambar blister lalu tekan DETEKSI", fg="blue", anchor="w")

    bawah = tk.Frame(root)
    bawah.pack(side="top", fill="both", expand=True)
    kanvas = tk.Canvas(bawah, bg="white")
    scroll_y = tk.Scrollbar(bawah, orient="vertical", command=kanvas.yview)
    kanvas.configure(yscrollcommand=scroll_y.set)
    scroll_y.pack(side="right", fill="y")
    kanvas.pack(side="left", fill="both", expand=True)
    isi_scroll = tk.Frame(kanvas, bg="white")
    kanvas.create_window((0, 0), window=isi_scroll, anchor="nw")

    def update_area_scroll(event):
        kanvas.configure(scrollregion=kanvas.bbox("all"))

    def scroll_mouse(event):
        kanvas.yview_scroll(int(-event.delta / 120), "units")

    isi_scroll.bind("<Configure>", update_area_scroll)
    root.bind_all("<MouseWheel>", scroll_mouse)

    def tempel_figure(fig):
        wadah = FigureCanvasTkAgg(fig, master=isi_scroll)
        wadah.draw()
        wadah.get_tk_widget().pack(side="top", pady=10)

    def pilih_gambar():
        p = filedialog.askopenfilename(title="Pilih gambar blister",
                                       filetypes=[("Gambar", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"), ("Semua file", "*.*")])
        if p != "":
            path_gambar[0] = p
            label_file.config(text="Gambar: " + p.split("/")[-1])

    def deteksi():
        if path_gambar[0] == "":
            status.config(text="Pilih gambar dulu!", fg="red")
            return
        for w in isi_scroll.winfo_children():
            w.destroy()
        status.config(text="Sedang memproses...", fg="blue")
        root.update()

        d = analisis(path_gambar[0])
        tempel_figure(figure_hasil(d))
        tempel_figure(figure_langkah(d))
        kanvas.yview_moveto(0)
        status.config(text="Selesai: sisa obat = " + str(d["jumlah"]) + "  (" +
                      str(round(d["waktu"], 2)) + " detik). Scroll untuk melihat langkahnya.", fg="green")

    tk.Button(atas, text="Pilih Gambar", command=pilih_gambar).grid(row=0, column=0, padx=4)
    label_file.grid(row=0, column=1)
    tk.Button(atas, text="DETEKSI", bg="#2e7d32", fg="white", width=12,
              command=deteksi).grid(row=0, column=2, padx=10)
    status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
    root.mainloop()


# ================================ MAIN =======================================
if len(sys.argv) > 1:
    for path in sys.argv[1:]:
        d = analisis(path)
        dipakai = [l[0] for l in d["langkah"] if l[3]]
        print(d["nama"], "-> sisa obat:", d["jumlah"], "| waktu", round(d["waktu"], 2), "s",
              "| perbaikan:", dipakai if len(dipakai) > 0 else "-",
              "| ditolak:", [x["alasan"] for x in d["ditolak"]])
else:
    jalankan_jendela()
