# =============================================================================
# deteksi.py  -  MENGHITUNG SISA OBAT DI KEMASAN BLISTER DARI FOTO HP
# Digital Image Processing murni: tanpa OpenCV / scipy / Machine Learning.
# Semua rumus ditulis manual di folder dip/ (satu file per materi slide).
#
# analisis(path) -> dict hasil (lihat kontrak: nama, jumlah, waktu, foto,
#                   obat, ditolak, langkah)
#
# Alur besar:
#   A. Foto utuh diperkecil (sampling) -> cari AREA BLISTER (ROI)
#   B. Blister dipotong dari foto asli -> perbaiki (restorasi + enhancement)
#   C. Cari calon obat di dalam blister -> region growing -> saring dengan
#      aturan bentuk & ukuran -> hitung -> gambar lingkaran
# =============================================================================
import math
import time
import numpy as np

from dip.dasar import baca_gambar, perkecil, faktor_sampling, potong_nilai, ke_abu, koordinat
from dip.m02_enhancement import (histogram, rata_dan_std, persentil, pakai_lut, lut_stretching,
                                 lut_gamma, lut_equalization, graylevel_slicing, threshold, ke_8bit)
from dip.m03_spasial import (konvolusi, kernel_mean, LOWPASS_32, filter_median3, per_kanal,
                             std_lokal)
from dip.m04_frekuensi import dft2, idft2, geser_tengah
from dip.m05_warna import (rgb_ke_hsi, minimum_rgb, jarak_warna, jarak_dua_warna, median_warna, warna_rata,
                           hsi_satu_warna, beda_hue, jarak_ke_abu)
from dip.m06_restorasi import (cek_salt_pepper, estimasi_noise, cek_periodik, butterworth_notch,
                               nilai_tengah)
from dip.m08_segmentasi import sobel, otsu, label_region, region_terbesar, region_growing
from dip.m09_representasi import (erosi, dilasi, opening, closing, isi_lubang, batas,
                                  mask_dari_piksel, chain_code, keliling, jari_jari, kebundaran, momen_region,
                                  jenis_bentuk)


# ================================ PARAMETER ==================================
UKURAN_KERJA     = 360    # sisi terpanjang foto utuh setelah sampling
UKURAN_OBAT      = 450    # sisi terpanjang potongan blister
MARGIN_POTONG    = 12     # px (gambar kecil) di sekeliling blister ikut dipotong

# restorasi & enhancement (06, 03, 02)
BATAS_SALT       = 0.01   # > 1% piksel ekstrem beda jauh dari median -> salt & pepper
BATAS_NOISE      = 6.0    # noise di patch datar > 6 -> lowpass
BATAS_PUNCAK     = 100    # titik spektrum > 100x median spektrum -> noise periodik
RADIUS_PUSAT     = 10     # frekuensi rendah (isi gambar) tidak dianggap noise
RADIUS_NOTCH     = 9      # D0 Butterworth notch
ORDE_NOTCH       = 2      # n Butterworth
BATAS_STD        = 40     # std keabuan < 40 -> kontras rendah -> stretching
BATAS_RENTANG    = 70     # rentang p1..p99 < 70 -> histogram sempit -> equalization
BATAS_GELAP      = 70     # rata-rata < 70 -> terlalu gelap -> gamma 0.5
BATAS_TERANG     = 185    # rata-rata > 185 -> terlalu terang -> gamma 2

# ROI blister (08, 09)
JENDELA_TEKSTUR  = 7      # lowpass 7x7 untuk skor foil & kepadatan tepi
CLOSING_ROI      = 13     # closing menyatukan foil blister
OPENING_ROI      = 17     # opening besar memutus silau meja yang menempel
PITA_TEPI_FOTO   = 6      # pita di tepi foto diabaikan

# kulit / tangan / meja kayu (05, 08, 09)
KULIT_H          = (5, 50)     # hue warna hangat (derajat): kulit, kayu
KULIT_S          = 0.12        # saturasi minimal (bukan abu-abu / foil)
KULIT_I          = 20          # tidak terlalu gelap
OPENING_KULIT    = 5           # buang bintik & putus jembatan tipis
PITA_KULIT       = 10          # region menyentuh pita 10 px di tepi foto
MIN_KULIT        = 0.02        # region > 2% foto
LEBAR_KULIT      = 5           # dilasi 5x5: pinggiran kulit ikut dibuang

# calon obat (03, 05, 08)
JENDELA_STD      = 5      # jendela std lokal (kehalusan)
SKALA_STD        = 40     # std 40 -> 255
BEDA_FOIL        = 40     # jarak warna ke foil > 40
BEDA_MEJA        = 45     # jarak warna ke meja < 45 -> kantong kosong / meja
BATAS_KILAU      = 235    # min(R,G,B) >= 235 = putih jenuh = pantulan cahaya
LEBAR_SILAU      = 7      # zona silau = dilasi 7x7 dari titik jenuh

# penilaian calon (09)
MIN_INTI         = 40     # region < 40 piksel diabaikan
MAKS_LUAS_ROI    = 0.15   # calon > 15% luas blister = foil datar
MIN_ISI          = 0.75   # luas / luas elips momen
MAKS_LONJONG     = 4.0    # sumbu panjang / sumbu pendek
LEBAR_CINCIN     = 6      # lebar cincin pembanding (px)
MIN_DIKELILINGI  = 0.85   # >= 85% cincin harus di dalam blister
BEDA_SEKITAR     = 30     # warna calon beda > 30 dari cincin sekitarnya
MIN_TAJAM        = 60     # rata-rata Sobel di garis batas (tepi tegas)
MAKS_CINCIN_MEJA = 0.40   # >= 40% cincin berwarna meja -> pantulan kantong kosong
BATAS_ABU        = 30     # jarak warna ke sumbu abu-abu < 30 = tidak berwarna
BEDA_HUE_MEJA    = 15     # hue calon beda < 15 derajat dari hue meja -> meja tembus plastik
MIN_S_HUE        = 0.10   # hue hanya berarti bila saturasi >= 0.10
PISAH_BERTINGKAT = (9, 13, 17)  # opening bertingkat memutus obat <-> foil datar

# region growing (08)
LANGKAH_TUMBUH   = 8      # maksimal 8 piksel ke luar
BATAS_TUMBUH     = 50     # piksel ikut bila jarak warna ke obat < 50
MAKS_TUMBUH      = 2.0    # tumbuh > 2x inti = bocor ke foil -> batal
OPENING_PISAH    = 11     # opening 11x11 memutus sambungan tipis antar region

# aturan akhir (09)
MAKS_KILAU       = 0.90   # region >= 90% di zona silau -> pantulan
MIN_LUAS_ROI     = 0.003  # obat minimal 0.3% luas blister
BATAS_KECIL      = 0.35   # luas < 0.35 x obat terbesar -> serpihan
BATAS_KECIL_MEDIAN = 0.5  # luas < 0.5 x median -> serpihan
BATAS_DOBEL      = 1.60   # luas >= 1.6 x median -> round(luas / median) butir
BEDA_DUA_WARNA   = 60     # 2 belahan beda warna > 60 -> kapsul
JARAK_KAPSUL     = 5      # celah minimum (px) yang boleh antara 2 belahan kapsul
KALI_CELAH_KAPSUL = 0.6   # celah maks = 0.6 x lebar belahan (bayangan/kilau di sambungan kapsul)
MIN_LONJONG_BELAHAN = 1.5  # region awal harus sudah lonjong untuk dilengkapi jadi kapsul
MAKS_BEDA_SUDUT  = 20     # sumbu kapsul hasil tumbuh searah region awal (derajat)
MAKS_KALI_MEDIAN = 2.5   # luas > 2.5 x median -> bukan obat (tangan / foil); 2 butir menempel ~2x
MIN_ISI_DOBEL    = 0.87   # 2 butir menempel hanya bila bentuk tetap kompak


# ============================== ALAT BANTU ===================================
def buat_langkah(judul, materi, teknik, rumus, deskripsi, dipakai, jenis, gambar,
                 hist=None, garis=None):
    return {"judul": judul, "materi": materi, "teknik": teknik, "rumus": rumus,
            "deskripsi": deskripsi, "dipakai": dipakai, "jenis": jenis, "gambar": gambar,
            "histogram": hist, "garis": garis}


def bulat(angka, digit):
    # angka -> teks dengan jumlah desimal tertentu
    if digit == 0:
        return str(int(round(angka)))
    return str(round(angka, digit))


def gambar_lingkaran(img, cx, cy, r, warna, tebal):
    # menggambar lingkaran: titik (cx + r cos t, cy + r sin t) diwarnai
    tinggi = img.shape[0]
    lebar = img.shape[1]
    langkah_sudut = 720
    for i in range(langkah_sudut):
        t = 2 * math.pi * i / langkah_sudut
        for d in range(tebal):
            x = int(round(cx + (r + d) * math.cos(t)))
            y = int(round(cy + (r + d) * math.sin(t)))
            if y >= 0 and y < tinggi and x >= 0 and x < lebar:
                img[y, x, 0] = warna[0]
                img[y, x, 1] = warna[1]
                img[y, x, 2] = warna[2]


# ================= LANGKAH PERBAIKAN (06 Restorasi, 02, 03, 04) ==============
def perbaiki_gambar(img):
    # Memeriksa kondisi gambar lalu MEMILIH sendiri filter yang perlu.
    # Hasil: gambar yang sudah diperbaiki + catatan setiap pemeriksaan.
    catatan = {}
    abu = ke_abu(img)

    # 1) salt & pepper -> median 3x3 (order-statistic filter)
    salt = cek_salt_pepper(abu)
    catatan["salt"] = salt
    catatan["median"] = salt > BATAS_SALT
    if catatan["median"]:
        img = per_kanal(img, filter_median3)
    abu = ke_abu(img)

    # 2) noise periodik -> DFT + Butterworth notch reject
    F, besar, puncak, tertinggi = cek_periodik(abu, RADIUS_PUSAT, BATAS_PUNCAK)
    catatan["spektrum"] = besar
    catatan["puncak"] = puncak
    catatan["tertinggi"] = tertinggi
    catatan["notch"] = len(puncak) > 0
    if catatan["notch"]:
        H = butterworth_notch(abu.shape[0], abu.shape[1], puncak, RADIUS_NOTCH, ORDE_NOTCH)
        for c in range(3):
            Fc = geser_tengah(dft2(img[:, :, c]))
            img[:, :, c] = idft2(geser_tengah(Fc * H)).real
        img = potong_nilai(img, 0, 255)
    abu = ke_abu(img)

    # 3) noise aditif diukur di patch datar -> lowpass (1/32)[1 3 1; 3 16 3; 1 3 1]
    noise = estimasi_noise(abu)
    catatan["noise"] = noise
    catatan["lowpass"] = noise > BATAS_NOISE
    if catatan["lowpass"]:
        img = per_kanal(img, konvolusi, LOWPASS_32)
    abu = ke_abu(img)
    catatan["abu_bersih"] = abu
    bersih = img.copy()        # bebas noise, intensitas masih asli

    # 4) intensitas: equalization / stretching / gamma (02 Enhancement)
    hist_awal = histogram(abu)
    rata, std = rata_dan_std(hist_awal)
    p1 = persentil(hist_awal, 0.01)
    p99 = persentil(hist_awal, 0.99)
    catatan["hist_awal"] = hist_awal
    catatan["rata"] = rata
    catatan["std"] = std
    catatan["p1"] = p1
    catatan["p99"] = p99
    catatan["equalization"] = (p99 - p1) < BATAS_RENTANG
    catatan["stretching"] = (not catatan["equalization"]) and std < BATAS_STD
    if catatan["equalization"]:
        img = per_kanal(img, pakai_lut, lut_equalization(hist_awal))
    if catatan["stretching"]:
        img = per_kanal(img, pakai_lut, lut_stretching(p1, p99))

    rata_baru, std_baru = rata_dan_std(histogram(ke_abu(img)))
    catatan["rata_baru"] = rata_baru
    catatan["gamma"] = 1.0
    if rata_baru < BATAS_GELAP:
        catatan["gamma"] = 0.5
    elif rata_baru > BATAS_TERANG:
        catatan["gamma"] = 2.0
    if catatan["gamma"] != 1.0:
        img = per_kanal(img, pakai_lut, lut_gamma(0, 255, catatan["gamma"]))
    catatan["hist_akhir"] = histogram(ke_abu(img))
    return img, bersih, catatan


def daftar_perbaikan(catatan):
    # nama filter yang dipakai (untuk ringkasan)
    nama = []
    if catatan["median"]:
        nama.append("median 3x3")
    if catatan["notch"]:
        nama.append("DFT + Butterworth notch")
    if catatan["lowpass"]:
        nama.append("lowpass 1/32")
    if catatan["equalization"]:
        nama.append("histogram equalization")
    if catatan["stretching"]:
        nama.append("contrast stretching")
    if catatan["gamma"] != 1.0:
        nama.append("gamma " + str(catatan["gamma"]))
    return nama


# ===================== CARI BLISTER / ROI (05, 08, 09) =======================
def cari_kulit(hue, saturasi, intensitas):
    # Tangan (kulit) dan meja kayu = warna HANGAT (H 5..50 derajat) dengan
    # saturasi sedang. Obat krem/oranye juga hangat, jadi warna saja tidak
    # cukup: dipakai juga STRUKTUR. Tangan/meja = region BESAR yang
    # menyentuh tepi foto (berlanjut ke luar blister). Obat selalu kecil
    # dan dikelilingi blister.
    tinggi = hue.shape[0]
    lebar = hue.shape[1]
    hangat = (hue >= KULIT_H[0]) * (hue <= KULIT_H[1])
    warna = hangat * (saturasi >= KULIT_S) * (intensitas >= KULIT_I)
    warna = opening(warna * 1.0, OPENING_KULIT)
    _, daftar = label_region(warna)
    kulit = np.zeros(hue.shape)
    p = PITA_KULIT
    for d in daftar:
        sentuh_tepi = d["y0"] < p or d["x0"] < p or d["y1"] >= tinggi - p or d["x1"] >= lebar - p
        if sentuh_tepi and d["luas"] > MIN_KULIT * hue.size:
            for (y, x) in d["piksel"]:
                kulit[y, x] = 1
    return warna, kulit


def cari_blister(img, saturasi, intensitas, kulit):
    # Foil blister = TERANG (I besar) + TIDAK BERWARNA (S kecil) + KASAR (banyak tepi).
    abu = ke_abu(img)
    tinggi = abu.shape[0]
    lebar = abu.shape[1]

    skor = konvolusi(intensitas * (1 - saturasi), kernel_mean(JENDELA_TEKSTUR))
    skor_8bit = ke_8bit(skor, skor.max())
    hist_skor = histogram(skor_8bit)
    batas_skor = otsu(hist_skor)

    tepi = sobel(abu)
    tekstur = konvolusi(tepi, kernel_mean(JENDELA_TEKSTUR))
    tekstur_8bit = tekstur * 255 / (tekstur.max() + 0.001)
    hist_tekstur = histogram(tekstur_8bit)
    batas_tekstur = otsu(hist_tekstur)

    foil = threshold(skor_8bit, batas_skor) * threshold(tekstur_8bit, batas_tekstur)
    foil = foil * (1 - kulit)              # kulit / meja bukan foil
    p = PITA_TEPI_FOTO                     # efek batas filter di tepi foto
    foil[:p, :] = 0
    foil[tinggi - p:, :] = 0
    foil[:, :p] = 0
    foil[:, lebar - p:] = 0

    roi = closing(foil, CLOSING_ROI)
    roi = isi_lubang(roi)
    roi = roi * (1 - kulit)                # jari yang menutupi blister dibuang lagi
    roi = opening(roi, OPENING_ROI)
    _, daftar_roi = label_region(roi)
    roi = region_terbesar(roi)
    ketemu = roi.sum() >= 0.01 * abu.size
    if not ketemu:                         # blister tidak ketemu -> seluruh foto
        roi = np.ones(abu.shape)
    return {"skor": skor_8bit, "hist_skor": hist_skor, "batas_skor": batas_skor,
            "tekstur": tekstur_8bit, "hist_tekstur": hist_tekstur, "batas_tekstur": batas_tekstur,
            "foil": foil, "roi": roi, "jumlah_region": len(daftar_roi), "ketemu": ketemu}


def potong_blister(asli, kecil, roi):
    # kotak pembatas ROI (gambar kecil) -> potong dari foto ASLI resolusi penuh,
    # lalu sampling lagi sampai sisi terpanjang <= UKURAN_OBAT.
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

    langkah = faktor_sampling(ay1 - ay0, ax1 - ax0, UKURAN_OBAT)
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
    return potong, roi_potong, langkah


def bungkus_oktagon(mask):
    # Blister persegi (boleh miring) tidak punya coakan. Mask dibungkus
    # oktagon: batas min/maks dari x, y, x+y, x-y (arah lurus & diagonal),
    # sehingga coakan karena bayangan / obat di tepi ikut tertutup.
    yy, xx = koordinat(mask.shape[0], mask.shape[1])
    ada = mask > 0.5
    hasil = np.ones(mask.shape)
    for nilai in (xx, yy, xx + yy, xx - yy):
        v = nilai[ada]
        hasil = hasil * (nilai >= v.min()) * (nilai <= v.max())
    return hasil


# ======================== PENILAIAN CALON OBAT ===============================
def potong_region(d, r, tinggi, lebar):
    # kotak di sekitar region (dengan margin r) + mask region di dalam kotak
    y0 = max(d["y0"] - r, 0)
    y1 = min(d["y1"] + r + 1, tinggi)
    x0 = max(d["x0"] - r, 0)
    x1 = min(d["x1"] + r + 1, lebar)
    m = np.zeros((y1 - y0, x1 - x0))
    for (y, x) in d["piksel"]:
        m[y - y0, x - x0] = 1
    return y0, y1, x0, x1, m


def luas_cincin_utuh(d, r):
    # luas cincin bila region TIDAK terpotong tepi gambar
    m = np.zeros((d["y1"] - d["y0"] + 1 + 2 * r + 2, d["x1"] - d["x0"] + 1 + 2 * r + 2))
    for (y, x) in d["piksel"]:
        m[y - d["y0"] + r + 1, x - d["x0"] + r + 1] = 1
    return (dilasi(m, 2 * r + 1) - dilasi(m, 5)).sum()


def nilai_calon(img, roi, d, luas_roi, jarak_meja, warna_meja):
    # Menilai satu calon obat TANPA melihat warna obat tertentu.
    # Hasil: "ok" atau alasan ditolak.
    if d["luas"] < MIN_INTI:
        return "kecil"
    if d["luas"] > MAKS_LUAS_ROI * luas_roi:
        return "besar"
    cy, cx, isi, lonjong = momen_region(d["piksel"])
    if isi < MIN_ISI or lonjong > MAKS_LONJONG:
        return "bentuk"

    # cincin di sekeliling calon (dilasi besar - dilasi kecil)
    r = LEBAR_CINCIN + 2
    y0, y1, x0, x1, m = potong_region(d, r, img.shape[0], img.shape[1])
    cincin = (dilasi(m, 2 * r + 1) - dilasi(m, 5)) * roi[y0:y1, x0:x1]
    # 1) harus dikelilingi blister
    if cincin.sum() < MIN_DIKELILINGI * luas_cincin_utuh(d, r):
        return "tepi"
    # 2) warnanya beda dengan cincin sekitarnya (obat menonjol dari kantongnya)
    potong = img[y0:y1, x0:x1]
    dalam = median_warna(potong, m)
    luar = median_warna(potong, cincin)
    if jarak_dua_warna(dalam, luar) < BEDA_SEKITAR:
        return "kontras"
    # 2b) lubang / bayangan: lebih GELAP dari sekitarnya dan TIDAK berwarna
    #     (foil sobek, celah gelap, bayangan). Obat putih selalu lebih terang,
    #     obat berwarna punya saturasi.
    h_dalam, s_dalam, i_dalam = hsi_satu_warna(dalam)
    h_luar, s_luar, i_luar = hsi_satu_warna(luar)
    if i_dalam < i_luar and jarak_ke_abu(dalam) < BATAS_ABU:
        return "lubang"
    # 2c) kantong kosong transparan: meja terlihat tembus plastik, jadi
    #     kecerahannya berubah tetapi HUE-nya tetap hue meja
    if warna_meja is not None:
        h_meja, s_meja, i_meja = hsi_satu_warna(warna_meja)
        if s_meja >= MIN_S_HUE and s_dalam >= MIN_S_HUE and s_dalam <= s_meja and beda_hue(h_dalam, h_meja) < BEDA_HUE_MEJA:
            return "tembus"
    # 3) tepi tegas: rata-rata Sobel di pita garis batas
    pita = dilasi(m, 7) - erosi(m, 5)
    tepi = sobel(ke_abu(potong))
    if (tepi * pita).sum() / (pita.sum() + 1) < MIN_TAJAM:
        return "pudar"
    # 4) kantong kosong transparan: di sekeliling kilau plastiknya terlihat meja
    if jarak_meja is not None:
        mirip_meja = (jarak_meja[y0:y1, x0:x1] < BEDA_MEJA) * cincin
        if mirip_meja.sum() >= MAKS_CINCIN_MEJA * cincin.sum():
            return "pantulan"
    return "ok"


def geser_region(b, y0, x0):
    # koordinat region dari kotak kecil -> koordinat gambar
    piksel_baru = []
    for (y, x) in b["piksel"]:
        piksel_baru.append((y + y0, x + x0))
    b["piksel"] = piksel_baru
    b["y0"] += y0
    b["y1"] += y0
    b["x0"] += x0
    b["x1"] += x0
    return b


def pecah_dan_nilai(img, roi, d, luas_roi, tingkat, jarak_meja, warna_meja, hitung):
    # Nilai region d. Bila gagal karena BENTUK / terlalu BESAR (obat menempel
    # ke foil datar), pecah dengan opening yang lebih besar lalu nilai lagi.
    alasan = nilai_calon(img, roi, d, luas_roi, jarak_meja, warna_meja)
    if alasan in hitung:
        hitung[alasan] += 1
    else:
        hitung[alasan] = 1
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
        lolos = lolos + pecah_dan_nilai(img, roi, geser_region(b, y0, x0), luas_roi,
                                        tingkat + 1, jarak_meja, warna_meja, hitung)
    return lolos


def tumbuhkan_obat(img, label, d, jarak_foil, jarak_meja, roi, terpakai):
    # REGION GROWING dari inti obat (seed). Piksel tetangga boleh ikut bila
    # warnanya mirip obat (jarak < BATAS_TUMBUH) dan lebih mirip obat
    # daripada foil / meja. Hasilnya mengikuti bentuk asli obat.
    tinggi = img.shape[0]
    lebar = img.shape[1]
    r = LANGKAH_TUMBUH
    y0 = max(d["y0"] - r, 0)
    y1 = min(d["y1"] + r + 1, tinggi)
    x0 = max(d["x0"] - r, 0)
    x1 = min(d["x1"] + r + 1, lebar)

    warna = warna_rata(img, d["piksel"])
    jarak_obat = jarak_warna(img[y0:y1, x0:x1], warna)
    boleh = (jarak_obat < jarak_foil[y0:y1, x0:x1]) * (jarak_obat < BATAS_TUMBUH)
    boleh = boleh * roi[y0:y1, x0:x1] * (1 - terpakai[y0:y1, x0:x1])
    if jarak_meja is not None:
        boleh = boleh * (jarak_obat < jarak_meja[y0:y1, x0:x1])

    inti = (label[y0:y1, x0:x1] == d["nomor"]) * 1.0
    tumbuh = region_growing(inti, boleh, r)
    tumbuh = isi_lubang(opening(closing(tumbuh, 5), 3))
    tumbuh = region_terbesar(tumbuh)
    bocor = False
    if tumbuh.sum() > MAKS_TUMBUH * d["luas"] or tumbuh.sum() == 0:
        bocor = True                         # bocor ke foil -> pakai inti saja
    else:
        _, daftar = label_region(tumbuh)
        _, _, isi, lonjong = momen_region(daftar[0]["piksel"])
        if isi < MIN_ISI:
            bocor = True                     # bentuk rusak -> inti saja
    if bocor:
        tumbuh = inti

    hasil = np.zeros((tinggi, lebar))
    hasil[y0:y1, x0:x1] = tumbuh
    return hasil, bocor


def pisahkan_sambungan(d, tinggi, lebar):
    # Opening 11x11 memutus "jembatan" tipis antara obat dan pantulan.
    # Bila region pecah jadi >= 2 bagian, tiap bagian dinilai sendiri.
    y0, y1, x0, x1, m = potong_region(d, 1, tinggi, lebar)
    _, bagian = label_region(opening(m, OPENING_PISAH))
    hasil = []
    for b in bagian:
        if b["luas"] >= MIN_INTI:
            hasil.append(geser_region(b, y0, x0))
    if len(hasil) >= 2:
        return hasil
    return [d]


def sudut_sumbu(piksel, cy, cx):
    # arah sumbu panjang region (radian) dari momen pusat orde 2
    m20 = 0
    m02 = 0
    m11 = 0
    for (y, x) in piksel:
        m20 += (y - cy) * (y - cy)
        m02 += (x - cx) * (x - cx)
        m11 += (y - cy) * (x - cx)
    return 0.5 * math.atan2(2 * m11, m02 - m20)


def warna_belahan(img, d):
    # Region dibelah dua tegak lurus sumbu panjangnya. Hasil: warna rata-rata
    # kedua belahan (None bila salah satu belahan kosong).
    cy = d["cy"]
    cx = d["cx"]
    sudut = sudut_sumbu(d["piksel"], cy, cx)
    ux = math.cos(sudut)
    uy = math.sin(sudut)
    kiri = []
    kanan = []
    for (y, x) in d["piksel"]:
        if (x - cx) * ux + (y - cy) * uy < 0:
            kiri.append((y, x))
        else:
            kanan.append((y, x))
    if len(kiri) == 0 or len(kanan) == 0:
        return None
    return warna_rata(img, kiri), warna_rata(img, kanan)


def cek_dua_warna(img, d):
    # Kapsul = dua warna: warna rata-rata kedua belahan dibandingkan dengan
    # jarak Euclid (materi 05).
    belahan = warna_belahan(img, d)
    if belahan is None:
        return False
    return jarak_dua_warna(belahan[0], belahan[1]) > BEDA_DUA_WARNA


def coba_gabung(a, b, img, tinggi, lebar):
    # Apakah region a dan b adalah DUA BELAHAN satu kapsul?
    #   1) bersentuhan / celah <= 0.6 x lebar belahan (dilasi a mengenai b)
    #   2) warnanya beda jelas (> BEDA_DUA_WARNA): belahan kapsul beda warna,
    #      sedangkan 2 obat sewarna yang menempel tetap dihitung terpisah
    #   3) gabungannya berbentuk kapsul: lonjong 1.6..4 dan isi elips >= 0.8
    # Hasil: region gabungan, atau None bila bukan satu kapsul.
    if jarak_dua_warna(warna_rata(img, a["piksel"]), warna_rata(img, b["piksel"])) <= BEDA_DUA_WARNA:
        return None
    # celah maksimum ikut ukuran obat: 0.6 x lebar belahan yang lebih besar.
    # lebar = sumbu pendek elips setara = 2 * sqrt(A / (pi * lonjong))
    besar = a
    if b["luas"] > a["luas"]:
        besar = b
    _, _, _, lonjong_besar = momen_region(besar["piksel"])
    lebar_belahan = 2 * math.sqrt(besar["luas"] / (math.pi * lonjong_besar))
    celah = int(KALI_CELAH_KAPSUL * lebar_belahan)
    if celah < JARAK_KAPSUL:
        celah = JARAK_KAPSUL
    r = celah + 1
    y0 = max(min(a["y0"], b["y0"]) - r, 0)
    y1 = min(max(a["y1"], b["y1"]) + r + 1, tinggi)
    x0 = max(min(a["x0"], b["x0"]) - r, 0)
    x1 = min(max(a["x1"], b["x1"]) + r + 1, lebar)
    ma = np.zeros((y1 - y0, x1 - x0))
    mb = np.zeros((y1 - y0, x1 - x0))
    for (y, x) in a["piksel"]:
        ma[y - y0, x - x0] = 1
    for (y, x) in b["piksel"]:
        mb[y - y0, x - x0] = 1
    if (dilasi(ma, 2 * celah + 1) * mb).sum() == 0:
        return None                              # tidak bersentuhan
    # celah di antara kedua belahan ditutup dengan closing, lalu isi lubang
    gabung = isi_lubang(closing(ma + mb - ma * mb, 2 * celah + 1))
    _, daftar = label_region(gabung)
    if len(daftar) != 1:
        return None
    c = geser_region(daftar[0], y0, x0)
    c["kapsul"] = True                          # tanda: hasil gabungan 2 belahan
    _, _, isi, lonjong = momen_region(c["piksel"])
    if isi < 0.8 or lonjong < 1.6 or lonjong > MAKS_LONJONG:
        return None
    return c


def gabung_kapsul(daftar, img, tinggi, lebar):
    # Gabungkan pasangan belahan kapsul berulang kali sampai tidak ada lagi.
    n_gabung = 0
    ada_gabung = True
    while ada_gabung:
        ada_gabung = False
        for i in range(len(daftar)):
            for j in range(i + 1, len(daftar)):
                c = coba_gabung(daftar[i], daftar[j], img, tinggi, lebar)
                if c is not None:
                    baru = []
                    for k in range(len(daftar)):
                        if k != i and k != j:
                            baru.append(daftar[k])
                    baru.append(c)
                    daftar = baru
                    n_gabung += 1
                    ada_gabung = True
                    break
            if ada_gabung:
                break
    return daftar, n_gabung


def lengkapi_kapsul(d, img, mirip_obat, sudah, tinggi, lebar, pakai_warna):
    # Kapsul 2 warna sering hanya tertangkap SEBAGIAN (satu belahan / satu
    # pita sempit), karena belahan kedua warnanya dekat foil atau meja.
    # Region digrowing lagi (materi 08) ke piksel "mirip obat" (halus ATAU
    # warnanya jauh dari foil & meja). Bila pakai_warna, piksel juga harus
    # dekat salah satu MODEL WARNA kapsul (warna belahan region + warna
    # belahan kedua yang diambil dari cincin di luar region).
    # Hasil hanya DIPAKAI bila menjadi kapsul: dua belahan beda warna,
    # lonjong 1.6..4, isi elips >= 0.8, dan sumbunya searah region awal.
    cy, cx, isi, lonjong = momen_region(d["piksel"])
    if lonjong < MIN_LONJONG_BELAHAN:
        return None                                 # bulat -> tablet, bukan belahan kapsul
    sumbu = math.sqrt(d["luas"] * lonjong / math.pi)     # setengah sumbu panjang
    r = int(2 * sumbu) + 2
    y0 = max(d["y0"] - r, 0)
    y1 = min(d["y1"] + r + 1, tinggi)
    x0 = max(d["x0"] - r, 0)
    x1 = min(d["x1"] + r + 1, lebar)
    asal = np.zeros((y1 - y0, x1 - x0))
    for (y, x) in d["piksel"]:
        asal[y - y0, x - x0] = 1
    boleh = mirip_obat[y0:y1, x0:x1] * (1 - sudah[y0:y1, x0:x1])

    if pakai_warna:
        kotak = img[y0:y1, x0:x1]
        d["cy"] = cy
        d["cx"] = cx
        model = [warna_rata(img, d["piksel"])]
        belahan = warna_belahan(img, d)
        if belahan is not None:
            model.append(belahan[0])
            model.append(belahan[1])
        jarak_model = jarak_warna(kotak, model[0])
        for w in model:
            jarak_model = np.minimum(jarak_model, jarak_warna(kotak, w))
        # cincin di luar region: piksel yang warnanya beda dari region; 20%
        # yang PALING beda (persentil 80) = warna belahan kedua
        cincin = (dilasi(asal, 9) - asal) * boleh
        lain = cincin * (jarak_model >= BATAS_TUMBUH)
        if lain.sum() >= 10:
            jarak_8bit = potong_nilai(jarak_model, 0, 255)
            t = persentil(histogram(jarak_8bit, lain), 0.8)
            jarak_model = np.minimum(jarak_model, jarak_warna(kotak, median_warna(kotak, lain * (jarak_8bit >= t))))
        boleh = boleh * (jarak_model < BATAS_TUMBUH)

    tumbuh = region_growing(asal, boleh, r)
    tumbuh = opening(tumbuh, 5)                     # putus jembatan tipis ke plastik
    # celah di sambungan kapsul (bayangan / kilau) ditutup closing ~0.5 x lebar
    k = int(math.sqrt(d["luas"] / (math.pi * lonjong))) // 2 * 2 + 1
    if k < 5:
        k = 5
    tumbuh = isi_lubang(closing(tumbuh + asal - tumbuh * asal, k))
    _, daftar = label_region(tumbuh)
    pilih = None
    for b in daftar:
        if pilih is None or b["luas"] > pilih["luas"]:
            pilih = b
    if pilih is None or pilih["luas"] <= d["luas"]:
        return None
    c = geser_region(pilih, y0, x0)
    c["cy"], c["cx"], c["isi"], c["lonjong"] = momen_region(c["piksel"])
    if c["isi"] < 0.8 or c["lonjong"] < 1.6 or c["lonjong"] > MAKS_LONJONG:
        return None
    beda = abs(sudut_sumbu(d["piksel"], cy, cx) - sudut_sumbu(c["piksel"], c["cy"], c["cx"]))
    beda = beda * 180 / math.pi
    if beda > 90:
        beda = 180 - beda
    if beda > MAKS_BEDA_SUDUT:
        return None                                 # tumbuh ke samping, bukan memanjang
    if not cek_dua_warna(img, c):
        return None
    c["kapsul"] = True
    return c


def segmen_lokal(img, cx, cy, r):
    # Region obat kadang hanya sebagian (sisi yang terkena cahaya), jadi untuk
    # NAMA BENTUK obat disegmentasi ulang di sekitar lingkarannya (materi 08):
    #   warna pembanding = median warna cincin r*1.15..r*1.35 (foil di sekitar obat)
    #   jarak warna setiap piksel ke pembanding -> threshold Otsu di dalam r*1.35
    #   ambil region terbesar di dalam r*1.2, isi lubangnya
    tinggi = img.shape[0]
    lebar = img.shape[1]
    R = int(r * 1.35) + 2
    y0 = max(int(cy) - R, 0)
    y1 = min(int(cy) + R + 1, tinggi)
    x0 = max(int(cx) - R, 0)
    x1 = min(int(cx) + R + 1, lebar)
    kotak = img[y0:y1, x0:x1]
    yy, xx = koordinat(y1 - y0, x1 - x0)
    jarak_pusat = ((yy + y0 - cy) ** 2 + (xx + x0 - cx) ** 2) ** 0.5
    cincin = (jarak_pusat > r * 1.15) * (jarak_pusat < r * 1.35)
    if cincin.sum() < 10:
        return None
    jarak = jarak_warna(kotak, median_warna(kotak, cincin))
    jarak_8bit = potong_nilai(jarak * 255 / (jarak.max() + 0.001), 0, 255)
    T = otsu(histogram(jarak_8bit, jarak_pusat < r * 1.35))
    mask = threshold(jarak_8bit, T) * (jarak_pusat < r * 1.2)
    mask = isi_lubang(opening(mask, 3))
    _, daftar = label_region(mask)
    terbesar = None
    for d in daftar:
        if terbesar is None or d["luas"] > terbesar["luas"]:
            terbesar = d
    if terbesar is None or terbesar["luas"] < MIN_INTI:
        return None
    piksel = []
    for (y, x) in terbesar["piksel"]:
        piksel.append((y + y0, x + x0))
    return piksel


def lengkapi_bentuk(d, img):
    # deskripsi region (materi 09): chain code, keliling, pusat, jari-jari, nama bentuk
    kode, titik = chain_code(d["piksel"])
    d["chain"] = kode
    d["keliling"] = keliling(kode)
    d["bundar"] = kebundaran(d["luas"], d["keliling"])
    d["r"] = jari_jari(titik, d["cy"], d["cx"]) + 1.5     # +1.5 px agar lingkaran menutup obat
    d["titik_batas"] = titik
    # nama bentuk dari outline hasil segmentasi ulang lokal (bila berhasil)
    piksel_bentuk = segmen_lokal(img, d["cx"], d["cy"], d["r"])
    if piksel_bentuk is None:
        piksel_bentuk = d["piksel"]
    cy, cx, isi, lonjong = momen_region(piksel_bentuk)
    _, titik_bentuk = chain_code(piksel_bentuk)
    seluruh = {"cy": cy, "cx": cx, "piksel": piksel_bentuk}
    if d.get("kapsul", False):
        d["bentuk"] = "kapsul"                   # gabungan 2 belahan beda warna
    else:
        d["bentuk"] = jenis_bentuk(titik_bentuk, cy, cx, lonjong, cek_dua_warna(img, seluruh))
    d["titik_bentuk"] = titik_bentuk
    return d


# ================================ ANALISIS ===================================
def analisis(path):
    mulai = time.time()
    langkah = []
    nama = path.replace("\\", "/").split("/")[-1]

    # ---------------------------------------------------------------- 1
    asli = baca_gambar(path)
    langkah.append(buat_langkah(
        "Foto asli", "01 Introduction to DIP I", "Citra digital f(x,y), 24 bit RGB",
        "f(x,y) = [R, G, B],  0 <= nilai <= 255",
        "Foto HP dibaca sebagai matriks " + str(asli.shape[0]) + " x " + str(asli.shape[1]) +
        " piksel dengan 3 kanal (R, G, B), masing-masing 8 bit (0..255). Foto sebesar ini terlalu "
        "berat untuk diproses dengan rumus manual, jadi langkah berikutnya memperkecilnya.",
        True, "rgb", asli))

    # ---------------------------------------------------------------- 2
    f_kecil = faktor_sampling(asli.shape[0], asli.shape[1], UKURAN_KERJA)
    kecil = perkecil(asli, UKURAN_KERJA)
    kecil = kecil[:kecil.shape[0] // 2 * 2, :kecil.shape[1] // 2 * 2]
    langkah.append(buat_langkah(
        "Perkecil foto (sampling)", "01 Introduction to DIP I", "Sampling (digitisasi koordinat)",
        "g(m,n) = f(f*m, f*n),  f = " + str(f_kecil),
        "Diambil 1 piksel setiap " + str(f_kecil) + " piksel sehingga foto menjadi " +
        str(kecil.shape[0]) + " x " + str(kecil.shape[1]) + ". Gambar kecil ini hanya dipakai untuk "
        "mencari letak blister. Sengaja tidak dirata-rata supaya noise salt & pepper tetap "
        "terlihat dan bisa dideteksi pada langkah restorasi.",
        True, "rgb", kecil))

    # ---------------------------------------------------------------- 3
    hue, sat, inten = rgb_ke_hsi(kecil)
    rata_s = sat.sum() / sat.size
    langkah.append(buat_langkah(
        "Ruang warna HSI (saturasi)", "05 Color Image Processing", "Konversi RGB -> HSI",
        "I = (R+G+B)/3;  S = 1 - 3 min(R,G,B)/(R+G+B);  H = theta (B<=G) / 360-theta",
        "Setiap piksel diubah ke H (hue), S (saturasi) dan I (intensitas) dengan rumus slide 05. "
        "Gambar menampilkan S: foil aluminium hampir tidak berwarna (S kecil, gelap di peta), "
        "sedangkan meja kayu / benda berwarna punya S besar. Rata-rata S seluruh foto = " +
        bulat(rata_s, 2) + ", rata-rata I = " + bulat(inten.sum() / inten.size, 0) +
        ". I dan S dipakai bersama untuk mencari foil blister.",
        True, "heat", sat))

    # ---------------------------------------------------------------- 4
    kecil_baik, kecil_bersih, cat_kecil = perbaiki_gambar(kecil.copy())
    pakai_kecil = []
    for n in daftar_perbaikan(cat_kecil):
        if n.startswith("median") or n.startswith("DFT") or n.startswith("lowpass"):
            pakai_kecil.append(n)
    if len(pakai_kecil) > 0:
        teks = "Filter yang dipakai: " + ", ".join(pakai_kecil) + "."
    else:
        teks = "Tidak ada noise yang perlu dibuang."
    langkah.append(buat_langkah(
        "Restorasi foto kecil", "06 Image Restoration", "Pemeriksaan noise otomatis",
        "g = f + eta;  median / notch / lowpass bila perlu",
        "Sebelum mencari blister, foto kecil diperiksa: salt & pepper " +
        bulat(cat_kecil["salt"] * 100, 2) + "% (batas " + bulat(BATAS_SALT * 100, 0) +
        "%), puncak spektrum " + bulat(cat_kecil["tertinggi"], 0) + "x median (batas " +
        str(BATAS_PUNCAK) + "), noise patch datar " + bulat(cat_kecil["noise"], 2) +
        " (batas " + str(BATAS_NOISE) + "). " + teks +
        " Kontras sengaja belum diubah karena equalization membuat meja gelap ikut terang.",
        len(pakai_kecil) > 0, "rgb", kecil_bersih))

    # ---------------------------------------------------------------- 5
    hue_bersih, sat_bersih, inten_bersih = rgb_ke_hsi(kecil_bersih)
    warna_kulit, kulit = cari_kulit(hue_bersih, sat_bersih, inten_bersih)
    blister = cari_blister(kecil_bersih, sat_bersih, inten_bersih, kulit)
    langkah.append(buat_langkah(
        "Skor foil: terang dan tidak berwarna", "08 Image Segmentation",
        "Lowpass 7x7 + threshold Otsu (Otsu di luar slide)",
        "skor = lowpass( I x (1 - S) );  foil bila skor > T",
        "Foil blister itu terang dan tidak berwarna, jadi skor = intensitas I x (1 - saturasi S) "
        "dari HSI (materi 05), "
        "dihaluskan dengan lowpass rata-rata 7x7 (materi 03). Histogram skor dibagi dua dengan "
        "T otomatis Otsu (cara memilih T dari Gonzalez, di luar slide) = " +
        str(blister["batas_skor"]) + " (skala 0..255). Meja kayu (berwarna) dan meja gelap "
        "mendapat skor kecil.",
        True, "hist", None, blister["hist_skor"], blister["batas_skor"]))

    # ---------------------------------------------------------------- 6
    langkah.append(buat_langkah(
        "Tekstur foil: kepadatan tepi Sobel", "08 Image Segmentation",
        "Deteksi tepi Sobel + lowpass + Otsu (di luar slide)",
        "grad f = |Gx| + |Gy|;  tekstur = lowpass 7x7(grad f)",
        "Foil berpola dan mengkilap sehingga banyak tepi kecil; silau di meja terang tetapi halus. "
        "Besar gradien Sobel dirata-rata 7x7 menjadi peta kepadatan tepi (gambar). "
        "Threshold Otsu pada peta ini = " + str(blister["batas_tekstur"]) +
        ". Piksel dianggap foil bila skor DAN tekstur sama-sama di atas threshold.",
        True, "heat", blister["tekstur"]))

    # ---------------------------------------------------------------- 7
    langkah.append(buat_langkah(
        "Area blister (ROI) dengan morfologi", "09 Representation & Description",
        "Closing, isi lubang (di luar slide), opening, region terbesar",
        "closing = dilasi lalu erosi (SE " + str(CLOSING_ROI) + "x" + str(CLOSING_ROI) +
        ");  opening = erosi lalu dilasi (SE " + str(OPENING_ROI) + "x" + str(OPENING_ROI) + ")",
        "Mask foil masih berlubang (kantong obat tidak bertekstur foil). Closing " +
        str(CLOSING_ROI) + "x" + str(CLOSING_ROI) + " menyatukan foil, lubang yang tidak "
        "tersambung ke tepi foto diisi (hole filling, Gonzalez). Daerah kulit/kayu (warna hangat, "
        "region besar menyentuh tepi foto: " + bulat(kulit.sum() * 100 / kulit.size, 0) + "% foto) "
        "dikeluarkan dari foil dan dari ROI agar jari yang menutupi blister tidak ikut, lalu opening " +
        str(OPENING_ROI) + "x" + str(OPENING_ROI) + " memutus silau meja yang menempel. "
        "Dari " + str(blister["jumlah_region"]) + " region (pelabelan N4) diambil yang terbesar"
        + (" sebagai blister." if blister["ketemu"] else "; tetapi terlalu kecil, jadi seluruh foto dipakai."),
        True, "mask", blister["roi"]))

    # ---------------------------------------------------------------- 8
    potong, roi_potong, f_potong = potong_blister(asli, kecil, blister["roi"])
    langkah.append(buat_langkah(
        "Potong blister dari foto asli", "01 Introduction to DIP I", "Cropping + sampling",
        "kotak ROI x " + bulat(asli.shape[0] / kecil.shape[0], 1) + " -> foto asli, sampling f = " +
        str(f_potong),
        "Kotak pembatas ROI dibesarkan ke koordinat foto asli (resolusi penuh), ditambah margin " +
        str(MARGIN_POTONG) + " px, lalu dipotong dan di-sampling 1 dari " + str(f_potong) +
        " piksel menjadi " + str(potong.shape[0]) + " x " + str(potong.shape[1]) + ". Obat jadi "
        "jauh lebih besar dibanding memakai foto kecil. Semua langkah berikutnya bekerja pada potongan ini.",
        True, "rgb", potong))

    # ------------------------------------------------------------ 9-12
    img, bersih, cat = perbaiki_gambar(potong.copy())
    langkah.append(buat_langkah(
        "Cek salt & pepper -> median filter", "06 Image Restoration",
        "Order-statistic filter: median 3x3 (materi 03)",
        "persen = #(piksel 0/255 yang |f - median| > 50) / n",
        "Noise impulse membuat piksel hitam/putih terisolasi. Dihitung piksel ekstrem (<= 8 atau "
        ">= 247) yang beda lebih dari 50 dari median 3x3 tetangganya: " +
        bulat(cat["salt"] * 100, 3) + "% (batas " + bulat(BATAS_SALT * 100, 0) + "%). " +
        ("Karena di atas batas, median 3x3 dijalankan pada tiap kanal." if cat["median"]
         else "Masih di bawah batas, jadi median filter dilewati agar detail tepi obat tidak hilang."),
        cat["median"], "gray", ke_abu(potong)))

    if cat["notch"]:
        teks = (str(len(cat["puncak"])) + " pasang titik noise ditemukan, jadi dibuang dengan "
                "Butterworth notch reject (D0 = " + str(RADIUS_NOTCH) + ", n = " + str(ORDE_NOTCH) +
                ") lalu DFT balik.")
    else:
        teks = "Tidak ada titik noise, filter notch dilewati."
    langkah.append(buat_langkah(
        "Cek noise periodik -> DFT + notch", "04 Frequency Domain + 06 Image Restoration",
        "DFT 2D, spektrum |F|, Butterworth notch reject",
        "F(u,v) = sum f(x,y) e^{-j2pi(ux/M+vy/N)};  H = 1/(1+[D0^2/(D1 D2)]^n)",
        "Noise periodik muncul sebagai titik terang di spektrum (gambar = log(1+|F|), F(0,0) di "
        "tengah). Titik terkuat di luar radius " + str(RADIUS_PUSAT) + " bernilai " +
        bulat(cat["tertinggi"], 0) + "x median spektrum (batas " + str(BATAS_PUNCAK) + "). " + teks,
        cat["notch"], "spektrum", cat["spektrum"]))

    langkah.append(buat_langkah(
        "Estimasi noise di patch datar -> lowpass", "06 Image Restoration",
        "Estimasi noise + mean filter (1/32)[1 3 1;3 16 3;1 3 1] (materi 03)",
        "residu = f - (f_atas + f_bawah + f_kiri + f_kanan)/4;  noise = rata |residu| di blok datar",
        "Gambar dibagi blok 8x8 dan dipilih blok yang paling datar (urutan 25%), supaya tepi "
        "obat tidak dihitung sebagai noise. Noise terukur = " + bulat(cat["noise"], 2) +
        " (batas " + str(BATAS_NOISE) + "). " +
        ("Di atas batas, jadi lowpass 1/32 dari slide dijalankan." if cat["lowpass"]
         else "Di bawah batas, jadi lowpass dilewati supaya tepi obat tetap tajam."),
        cat["lowpass"], "gray", cat["abu_bersih"]))

    perbaikan = []
    if cat["equalization"]:
        perbaikan.append("histogram equalization (rentang p1..p99 = " + str(cat["p99"] - cat["p1"]) +
                         " < " + str(BATAS_RENTANG) + ")")
    if cat["stretching"]:
        perbaikan.append("contrast stretching [" + str(cat["p1"]) + ".." + str(cat["p99"]) +
                         "] -> [0..255] karena std " + bulat(cat["std"], 1) + " < " + str(BATAS_STD))
    if cat["gamma"] != 1.0:
        perbaikan.append("gamma g = " + str(cat["gamma"]) + " karena rata-rata " +
                         bulat(cat["rata_baru"], 0))
    if len(perbaikan) > 0:
        teks = "Dipakai: " + "; ".join(perbaikan) + "."
    else:
        teks = "Histogram sudah lebar dan terang cukup, jadi tidak diubah."
    langkah.append(buat_langkah(
        "Histogram & perbaikan kontras", "02 Image Enhancement",
        "Histogram, equalization / contrast stretching / gamma",
        "p(r_k) = n_k/n;  s_k = sum p(r_j);  stretching (r1,0)-(r2,255);  T(r) = ((r-r1)/(r2-r1))^g",
        "Histogram potongan: rata-rata " + bulat(cat["rata"], 0) + ", std " + bulat(cat["std"], 1) +
        ", 1% tergelap <= " + str(cat["p1"]) + ", 1% terterang >= " + str(cat["p99"]) + ". " + teks +
        " Garis = rata-rata keabuan.",
        len(perbaikan) > 0, "hist", None, cat["hist_awal"], int(round(cat["rata"]))))

    # ---------------------------------------------------------------- 13
    # pantulan cahaya: putih jenuh di ketiga kanal (graylevel slicing pada min(R,G,B))
    kilau = graylevel_slicing(minimum_rgb(bersih), BATAS_KILAU, 255)
    zona_silau = dilasi(kilau, LEBAR_SILAU)
    langkah.append(buat_langkah(
        "Zona silau (pantulan cahaya)", "02 Image Enhancement + 09 Morfologi",
        "Graylevel slicing + dilasi",
        "silau = 1 bila min(R,G,B) >= " + str(BATAS_KILAU) + ";  zona = dilasi " +
        str(LEBAR_SILAU) + "x" + str(LEBAR_SILAU),
        "Pantulan lampu di plastik/foil membuat ketiga kanal jenuh. Graylevel slicing rentang [" +
        str(BATAS_KILAU) + "..255] pada min(R,G,B) menandai " + bulat(kilau.sum(), 0) +
        " piksel, lalu dilasi " + str(LEBAR_SILAU) + "x" + str(LEBAR_SILAU) +
        " ikut menandai pinggiran pantulan. Region yang >= " + bulat(MAKS_KILAU * 100, 0) +
        "% berada di zona ini nanti ditolak sebagai pantulan.",
        True, "mask", zona_silau))

    # ---------------------------------------------------------- 13b
    # tangan / jari: warna hangat + region besar yang menyentuh tepi potongan
    # (berlanjut ke luar blister). Obat tidak pernah menyentuh tepi potongan.
    hue_p, sat_p, inten_p = rgb_ke_hsi(bersih)
    hangat_p, kulit_p = cari_kulit(hue_p, sat_p, inten_p)
    kulit_p = dilasi(kulit_p, LEBAR_KULIT)
    roi_awal = roi_potong
    roi_potong = roi_potong * (1 - kulit_p)
    langkah.append(buat_langkah(
        "Buang tangan / jari (kulit)", "05 Color Image Processing + 08 Segmentation + 09 Morfologi",
        "HSI threshold, opening, pelabelan region, dilasi",
        "kulit = (" + str(KULIT_H[0]) + " <= H <= " + str(KULIT_H[1]) +
        ") AND S >= " + str(KULIT_S) + " AND I >= " + str(KULIT_I) +
        ";  dipakai bila region > " + bulat(MIN_KULIT * 100, 0) + "% dan menyentuh tepi",
        "Kulit tangan dan kayu berwarna hangat (hue " + str(KULIT_H[0]) + ".." + str(KULIT_H[1]) +
        " derajat) dengan saturasi >= " + str(KULIT_S) +
        ". Obat krem/oranye juga hangat, jadi dipakai juga STRUKTUR: setelah opening " +
        str(OPENING_KULIT) + "x" + str(OPENING_KULIT) + ", hanya region yang luasnya > " +
        bulat(MIN_KULIT * 100, 0) + "% potongan DAN menyentuh pita " + str(PITA_KULIT) +
        " px di tepi potongan yang dianggap tangan/meja (obat selalu di dalam blister, tidak sampai "
        "tepi). Region itu didilasi " + str(LEBAR_KULIT) + "x" + str(LEBAR_KULIT) +
        " lalu dikeluarkan dari ROI: " + bulat((roi_awal * kulit_p).sum() * 100 / (roi_awal.sum() + 1), 1) +
        "% ROI dibuang.",
        True, "mask", kulit_p))

    # ---------------------------------------------------------------- 14
    roi = roi_potong
    abu = ke_abu(img)
    std = std_lokal(abu, JENDELA_STD)
    std_8bit = potong_nilai(std * 255 / SKALA_STD, 0, 255)
    batas_std = otsu(histogram(std_8bit))
    halus = (std_8bit < batas_std) * roi
    kasar = (std_8bit >= batas_std) * roi
    langkah.append(buat_langkah(
        "Kehalusan permukaan (std lokal)", "03 Spatial Filtering + 08 Segmentation",
        "Dua lowpass rata-rata 5x5 -> simpangan baku lokal, threshold Otsu (di luar slide)",
        "std = sqrt( mean(f^2) - mean(f)^2 )  di jendela " + str(JENDELA_STD) + "x" + str(JENDELA_STD),
        "Permukaan obat halus, foil aluminium kasar (banyak kilau kecil). Simpangan baku lokal "
        "dihitung dari lowpass f dan f^2, diskalakan (std " + str(SKALA_STD) + " -> 255), lalu "
        "dibagi dengan Otsu: T = " + str(batas_std) + ". Di dalam blister " +
        bulat(halus.sum() * 100 / (roi.sum() + 1), 0) + "% piksel halus (calon obat) dan sisanya kasar (foil).",
        True, "heat", std_8bit))

    # ---------------------------------------------------------------- 15
    warna_foil = median_warna(img, kasar)
    jarak_foil = jarak_warna(img, warna_foil)
    luar = 1 - dilasi(roi, 9)
    warna_meja = None
    jarak_meja = None
    if luar.sum() > 0.03 * abu.size:
        warna_meja = median_warna(img, luar)
        jarak_meja = jarak_warna(img, warna_meja)

    inti = halus * (jarak_foil > BEDA_FOIL)
    if jarak_meja is not None:
        inti = inti * (jarak_meja > BEDA_MEJA)
    tepi = sobel(abu)
    tepi_skala = tepi * 255 / (tepi.max() + 0.001)
    p99 = persentil(histogram(tepi_skala), 0.99)          # 1% tepi terkuat = 255
    tepi_8bit = potong_nilai(tepi_skala * 255 / (p99 + 1), 0, 255)
    batas_tepi = otsu(histogram(tepi_8bit, roi))
    inti = inti * (tepi_8bit < batas_tepi)
    inti = opening(inti, 5)
    if warna_meja is not None:
        teks_meja = ("warna meja (median di luar blister) = " + str(warna_meja) +
                     ", piksel yang jaraknya < " + str(BEDA_MEJA) + " dibuang (kantong kosong menampakkan meja)")
    else:
        teks_meja = "meja hampir tidak terlihat, jadi cek warna meja dilewati"
    langkah.append(buat_langkah(
        "Calon obat: halus & beda warna dari foil", "05 Color + 08 Segmentation",
        "Jarak warna Euclid, threshold, Sobel, opening 5x5",
        "calon = halus AND De(warna, foil) > " + str(BEDA_FOIL) + " AND grad f < T_tepi",
        "Warna foil = median RGB daerah kasar = " + str(warna_foil) + ". Calon obat adalah piksel "
        "halus yang jarak Euclid warnanya ke foil > " + str(BEDA_FOIL) + " (obat putih, kuning, "
        "merah... semua bisa); " + teks_meja + ". Piksel di tepi Sobel kuat (T Otsu = " +
        str(batas_tepi) + ") dibuang agar obat terpisah dari pantulan, lalu opening 5x5 membuang bintik.",
        True, "mask", inti))

    # ---------------------------------------------------------------- 16
    luas_roi = roi.sum()
    roi_okta = bungkus_oktagon(roi) * (1 - kulit_p)
    label_calon, daftar_calon = label_region(inti)
    hitung = {}
    inti_lolos = np.zeros(abu.shape)
    for d in daftar_calon:
        if d["luas"] < MIN_INTI:
            continue
        for b in pecah_dan_nilai(img, roi_okta, d, luas_roi, 0, jarak_meja, warna_meja, hitung):
            for (y, x) in b["piksel"]:
                inti_lolos[y, x] = 1
    inti_lolos = opening(isi_lubang(closing(inti_lolos, 9)), 5)
    teks = []
    for alasan in hitung:
        teks.append(alasan + " " + str(hitung[alasan]))
    langkah.append(buat_langkah(
        "Penilaian setiap calon (seed)", "09 Representation & Description",
        "Luas, momen (di luar slide), cincin dilasi, HSI calon vs cincin vs meja, opening bertingkat",
        "cincin = dilasi(A, " + str(2 * LEBAR_CINCIN + 5) + ") - dilasi(A, 5);  beda warna dalam vs cincin > " +
        str(BEDA_SEKITAR) + ";  lubang: I_dalam < I_cincin dan jarak ke abu-abu < " + str(BATAS_ABU) +
        ";  tembus: |H_dalam - H_meja| < " + str(BEDA_HUE_MEJA),
        "Setiap calon dinilai: luas wajar, bentuk kompak (momen orde 2, Gonzalez), >= " +
        bulat(MIN_DIKELILINGI * 100, 0) + "% cincinnya di dalam blister, warnanya beda > " +
        str(BEDA_SEKITAR) + " dari cincin, tepi Sobel tegas (> " + str(MIN_TAJAM) + "), dan "
        "cincinnya bukan warna meja. Dengan HSI (materi 05) median warna calon dibandingkan: "
        "calon yang LEBIH GELAP (I) dari cincinnya dan tidak berwarna (jarak RGB ke sumbu abu-abu "
        "(I,I,I) < " + str(BATAS_ABU) + ") adalah lubang foil sobek / bayangan / celah gelap "
        "('lubang'); calon yang hue H-nya beda < " + str(BEDA_HUE_MEJA) + " derajat dari hue meja "
        "(keduanya S >= " + str(MIN_S_HUE) + ") adalah kantong kosong yang menampakkan meja tembus "
        "plastik - plastik hanya mengubah kecerahan, bukan hue ('tembus'). Calon yang menempel ke foil dipecah dengan opening " +
        str(PISAH_BERTINGKAT) + ". Hasil penilaian: " + ", ".join(teks) + ".",
        True, "mask", inti_lolos))

    # ---------------------------------------------------------------- 17
    label_inti, daftar_inti = label_region(inti_lolos)
    obat_mask = np.zeros(abu.shape)
    n_bocor = 0
    n_seed = 0
    for d in daftar_inti:
        if d["luas"] < MIN_INTI:
            continue
        n_seed += 1
        tumbuh, bocor = tumbuhkan_obat(img, label_inti, d, jarak_foil, jarak_meja, roi, obat_mask)
        if bocor:
            n_bocor += 1
        obat_mask = obat_mask + tumbuh
        obat_mask[obat_mask > 1] = 1
    obat_mask = isi_lubang(closing(obat_mask, 5))     # kapsul 2 warna disatukan lagi
    langkah.append(buat_langkah(
        "Region growing dari setiap seed", "08 Image Segmentation", "Region growing (N8)",
        "q ikut region bila De(f(q), warna seed) < " + str(BATAS_TUMBUH) +
        " dan lebih mirip obat daripada foil/meja",
        "Dari " + str(n_seed) + " seed, region dilebarkan 1 piksel per putaran (maks " +
        str(LANGKAH_TUMBUH) + " putaran) ke tetangga N8 yang warnanya mirip rata-rata seed. "
        "Hasilnya mengikuti bentuk asli obat (bulat, lonjong, kapsul). " + str(n_bocor) +
        " region bocor ke foil (> " + bulat(MAKS_TUMBUH, 0) + "x seed atau bentuk rusak) sehingga "
        "kembali ke seed-nya.",
        True, "mask", obat_mask))

    # ---------------------------------------------------------------- 18
    tinggi = abu.shape[0]
    lebar = abu.shape[1]
    _, daftar_awal = label_region(obat_mask, 4)
    daftar = []
    for d in daftar_awal:
        if d["luas"] >= MIN_INTI:
            daftar = daftar + pisahkan_sambungan(d, tinggi, lebar)
    # belahan kapsul yang beda warna (mis. merah | putih) disatukan -> 1 butir
    daftar, n_gabung = gabung_kapsul(daftar, potong, tinggi, lebar)
    # kapsul yang baru tertangkap sebagian dilengkapi (region growing lagi)
    mirip_obat = (jarak_foil > BEDA_FOIL) * 1.0
    if jarak_meja is not None:
        mirip_obat = mirip_obat * (jarak_meja > BEDA_MEJA)
    halus_semua = (std_8bit < batas_std) * (tepi_8bit < batas_tepi)
    mirip_obat = (halus_semua + mirip_obat - halus_semua * mirip_obat) * roi_okta
    mirip_obat = opening(mirip_obat, 5)             # bintik foil dibuang
    # region diproses dari yang terbesar; piksel kapsul yang sudah jadi
    # tidak boleh diambil region lain
    urut = []
    for d in daftar:
        urut.append((d["luas"], len(urut), d))
    urut.sort(reverse=True)
    sudah = np.zeros((tinggi, lebar))
    baru = []
    n_lengkap = 0
    for (_, _, d) in urut:
        n_dalam = 0
        for (y, x) in d["piksel"]:
            n_dalam += sudah[y, x]
        if n_dalam > 0.5 * d["luas"]:
            continue                                # sudah termasuk kapsul lain
        c = None
        n_kilau = 0
        for (y, x) in d["piksel"]:
            n_kilau += zona_silau[y, x]
        # hanya region yang bukan pantulan dan tidak terlalu kecil
        layak = n_kilau < MAKS_KILAU * d["luas"] and d["luas"] >= MIN_LUAS_ROI * roi.sum()
        if layak and not d.get("kapsul", False):
            c = lengkapi_kapsul(d, potong, mirip_obat, sudah, tinggi, lebar, False)
            if c is None:
                c = lengkapi_kapsul(d, potong, mirip_obat, sudah, tinggi, lebar, True)
        if c is not None:
            d = c
            n_lengkap += 1
        for (y, x) in d["piksel"]:
            sudah[y, x] = 1
        baru.append(d)
    daftar = baru

    calon = []
    ditolak = []
    for d in daftar:
        d["cy"], d["cx"], d["isi"], d["lonjong"] = momen_region(d["piksel"])
        n_kilau = 0
        for (y, x) in d["piksel"]:
            n_kilau += zona_silau[y, x]
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

    # aturan ukuran: obat dalam satu blister sama besar
    obat = []
    jumlah = 0
    median_luas = 0
    if len(calon) > 0:
        # patokan ukuran = median luas calon (bukan yang terbesar: tangan /
        # foil yang ikut terdeteksi bisa jauh lebih besar dari obat)
        semua_luas = []
        for d in calon:
            semua_luas.append(d["luas"])
        patokan = nilai_tengah(semua_luas)
        sisa = []
        for d in calon:
            if d["luas"] < BATAS_KECIL * patokan:
                d["alasan"] = "kecil"
                ditolak.append(d)
            elif d["luas"] > MAKS_KALI_MEDIAN * patokan and not d.get("kapsul", False):
                d["alasan"] = "besar"
                ditolak.append(d)
            else:
                sisa.append(d)
        if len(sisa) > 0:
            semua_luas = []
            for d in sisa:
                semua_luas.append(d["luas"])
            median_luas = nilai_tengah(semua_luas)
            for d in sisa:
                if d["luas"] < BATAS_KECIL_MEDIAN * median_luas:
                    d["alasan"] = "kecil"
                    ditolak.append(d)
                    continue
                d["butir"] = 1
                # kapsul gabungan 2 belahan = tetap 1 butir walaupun luasnya besar
                if (d["luas"] >= BATAS_DOBEL * median_luas and d["isi"] >= MIN_ISI_DOBEL
                        and not d.get("kapsul", False)):
                    d["butir"] = int(round(d["luas"] / median_luas))
                jumlah += d["butir"]
                obat.append(d)

    for d in obat + ditolak:
        lengkapi_bentuk(d, potong)

    # gambar batas (chain code) di atas potongan: hijau = obat, merah = ditolak
    gambar_batas = potong_nilai(potong, 0, 255) * 0.6
    for d in ditolak:
        # region ditolak: batas = A - erosi(A), diwarnai merah
        garis = batas(mask_dari_piksel(d["piksel"], tinggi, lebar))
        for c in range(3):
            kanal = gambar_batas[:, :, c]
            kanal[garis > 0.5] = [255, 60, 60][c]
    for d in obat:
        for (y, x) in d["titik_batas"]:
            gambar_batas[y, x] = [60, 255, 90]
        y, x = d["titik_batas"][0]
        gambar_batas[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = [255, 255, 0]
    teks_obat = []
    nomor = 0
    for d in obat:
        nomor += 1
        teks_obat.append("#" + str(nomor) + " (" + d["bentuk"] + "): A=" + str(d["luas"]) + ", P=" + bulat(d["keliling"], 0) +
                         ", 4piA/P^2=" + bulat(d["bundar"], 2) + ", r=" + bulat(d["r"], 0) +
                         (", " + str(d["butir"]) + " butir" if d["butir"] > 1 else ""))
    teks_tolak = []
    for d in ditolak:
        teks_tolak.append(d["alasan"])
    langkah.append(buat_langkah(
        "Pelabelan N4, batas & chain code, aturan", "09 Representation & Description + 02 Intro II",
        "Label N4, chain code 8 arah, keliling, centroid, jari-jari De, momen (di luar slide)",
        "P = #genap + sqrt(2) #ganjil;  pusat = rata-rata (x, y);  r = maks De(pusat, batas);  "
        "butir = round(A / median A) bila A >= " + str(BATAS_DOBEL) + " median",
        "Region diberi label (N4). Dua region yang berdekatan (celah <= " + str(KALI_CELAH_KAPSUL) +
        " x lebar belahan), warnanya beda > " + str(BEDA_DUA_WARNA) + " dan gabungannya berbentuk kapsul (lonjong 1.6.." +
        str(MAKS_LONJONG) + ", isi elips >= 0.8) adalah dua belahan SATU kapsul, jadi disatukan dengan "
        "closing: " + str(n_gabung) + " kapsul disatukan. Kapsul yang baru tertangkap sebagian (region lonjong >= " +
        str(MIN_LONJONG_BELAHAN) + ") digrowing lagi ke piksel mirip obat (halus, atau warnanya jauh dari foil " +
        "dan meja); bila perlu hanya ke piksel yang dekat warna belahan region atau warna belahan kedua (20% piksel " +
        "cincin yang paling beda warna). Hasil dipakai hanya bila berbentuk kapsul dua warna (lonjong 1.6.." +
        str(MAKS_LONJONG) + ", isi >= 0.8, sumbu bergeser <= " + str(MAKS_BEDA_SUDUT) + " derajat): " +
        str(n_lengkap) + " kapsul dilengkapi. 4piA/P^2 (kebundaran, Gonzalez) hanya informasi, tidak dipakai menyaring. Batas obat (hijau) ditelusuri dengan chain code 8 arah (titik kuning = "
        "awal), batas region ditolak (merah) = A - erosi(A). "
        "Aturan: tolak bila >= " + bulat(MAKS_KILAU * 100, 0) + "% di zona silau, luas < " +
        bulat(MIN_LUAS_ROI * 100, 1) + "% blister, isi elips < " + str(MIN_ISI) + ", lonjong > " +
        str(MAKS_LONJONG) + ", atau jauh lebih kecil dari obat lain. Median luas obat = " +
        str(median_luas) + " px. " +
        ("Diterima: " + "; ".join(teks_obat) + ". " if len(teks_obat) > 0 else "Tidak ada region yang diterima. ") +
        ("Ditolak: " + ", ".join(teks_tolak) + "." if len(teks_tolak) > 0 else "Tidak ada yang ditolak."),
        True, "rgb", gambar_batas))

    # ---------------------------------------------------------------- 19
    gambar_akhir = potong_nilai(potong, 0, 255)
    for d in obat:
        gambar_lingkaran(gambar_akhir, d["cx"], d["cy"], d["r"], [40, 230, 80], 3)
    # hitung jumlah obat per bentuk
    per_bentuk = {}
    for d in obat:
        if d["bentuk"] in per_bentuk:
            per_bentuk[d["bentuk"]] += d["butir"]
        else:
            per_bentuk[d["bentuk"]] = d["butir"]
    teks_bentuk = []
    for nama_bentuk in per_bentuk:
        teks_bentuk.append(nama_bentuk + " " + str(per_bentuk[nama_bentuk]))
    langkah.append(buat_langkah(
        "Hasil: lingkaran & bentuk setiap sisa obat", "09 Representation & Description",
        "Lingkaran dari centroid & jari-jari; bentuk dari momen, signature & warna",
        "x = cx + r cos t,  y = cy + r sin t;  signature r(i) = De(pusat, batas ke-i);  sudut = puncak r(i)",
        "Setiap obat diberi lingkaran berpusat di centroid dengan jari-jari = jarak Euclid terjauh "
        "dari pusat ke batasnya, sehingga lingkaran menutupi seluruh obat. Nama bentuk: lonjong >= 1.35 "
        "(momen) -> kapsul bila kedua belahannya beda warna > " + str(BEDA_DUA_WARNA) + ", selain itu oval / "
        "kaplet; signature (jarak pusat ke batas sepanjang chain code) hampir datar -> bulat; jumlah "
        "puncak signature = jumlah sudut (3 segitiga, 4 persegi, ...). Jumlah region obat = " +
        str(len(obat)) + ", total butir = " + str(jumlah) + " (region seluas ~2x median dihitung 2 butir)" +
        (". Per bentuk: " + ", ".join(teks_bentuk) + "." if len(teks_bentuk) > 0 else "."),
        True, "rgb", gambar_akhir))

    hasil_obat = []
    for d in obat:
        hasil_obat.append({"cx": d["cx"], "cy": d["cy"], "r": d["r"], "butir": d["butir"],
                           "bentuk": d["bentuk"]})
    hasil_tolak = []
    for d in ditolak:
        hasil_tolak.append({"cx": d["cx"], "cy": d["cy"], "r": d["r"], "alasan": d["alasan"]})

    return {"nama": nama, "jumlah": jumlah, "waktu": time.time() - mulai,
            "foto": potong_nilai(potong, 0, 255), "obat": hasil_obat, "ditolak": hasil_tolak,
            "langkah": langkah}
