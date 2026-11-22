# =============================================================================
# m09_representasi.py  -  Materi 09 Representation & Description
#   Morfologi (structuring element / SE kotak k x k, semua bernilai 1):
#     - dilasi : hasil 1 bila MINIMAL SATU posisi SE mengenai piksel 1 (OR/maks)
#     - erosi  : hasil 1 bila SEMUA posisi SE mengenai piksel 1 (AND/min)
#     - opening = erosi lalu dilasi  (buang bintik kecil, putus jembatan tipis)
#     - closing = dilasi lalu erosi  (tutup celah / lubang kecil)
#   Representasi batas:
#     - chain code 8 arah:   3 2 1
#                            4 x 0
#                            5 6 7
#     - keliling dari chain code: kode genap = 1 langkah, kode ganjil = sqrt(2)
#   Deskripsi region: luas, pusat (centroid), jari-jari, kebundaran.
# =============================================================================
import math
import numpy as np

from dip.m03_spasial import filter_minimum, filter_maksimum
from dip.m08_segmentasi import label_region


def erosi(mask, k):
    # erosi dengan SE kotak k x k  (= filter minimum, materi 03)
    return filter_minimum(mask, k)


def dilasi(mask, k):
    # dilasi dengan SE kotak k x k  (= filter maksimum, materi 03)
    return filter_maksimum(mask, k)


def opening(mask, k):
    return dilasi(erosi(mask, k), k)


def closing(mask, k):
    return erosi(dilasi(mask, k), k)


def isi_lubang(mask):
    # [DI LUAR SLIDE - Gonzalez] Hole filling: lubang = daerah 0 yang
    # TIDAK tersambung ke tepi gambar -> dijadikan 1.
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


def batas(mask):
    # garis batas = A - erosi(A)   (rumus Gonzalez, dibangun dari erosi slide)
    return mask - erosi(mask, 3)


def mask_dari_piksel(piksel, tinggi, lebar):
    m = np.zeros((tinggi, lebar))
    for (y, x) in piksel:
        m[y, x] = 1
    return m


# arah chain code 8 (dy, dx):  0 kanan, 1 kanan-atas, 2 atas, ... 7 kanan-bawah
ARAH_8 = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]


def chain_code(piksel):
    # Menelusuri batas luar region berlawanan arah jarum jam, mulai dari piksel
    # paling atas-kiri. Setiap langkah ke tetangga N8 dicatat kodenya (0..7).
    isi = {}
    for p in piksel:
        isi[p] = True
    mulai = piksel[0]
    for p in piksel:
        if p[0] < mulai[0] or (p[0] == mulai[0] and p[1] < mulai[1]):
            mulai = p

    kode = []
    titik = [mulai]
    sekarang = mulai
    arah = 7                       # arah terakhir (awal: anggap datang dari kiri-bawah)
    batas_langkah = 4 * len(piksel) + 8
    while len(kode) < batas_langkah:
        # mulai mencari dari arah "belakang-kiri" arah terakhir
        if arah % 2 == 0:
            cari = (arah + 7) % 8
        else:
            cari = (arah + 6) % 8
        ketemu = False
        for i in range(8):
            k = (cari + i) % 8
            ny = sekarang[0] + ARAH_8[k][0]
            nx = sekarang[1] + ARAH_8[k][1]
            if (ny, nx) in isi:
                kode.append(k)
                sekarang = (ny, nx)
                titik.append(sekarang)
                arah = k
                ketemu = True
                break
        if not ketemu:             # region 1 piksel
            break
        if sekarang == mulai:
            break
    return kode, titik


def keliling(kode):
    # keliling = (#kode genap) * 1 + (#kode ganjil) * sqrt(2)
    genap = 0
    ganjil = 0
    for k in kode:
        if k % 2 == 0:
            genap += 1
        else:
            ganjil += 1
    return genap + ganjil * math.sqrt(2)


def pusat_region(piksel):
    # centroid = rata-rata koordinat piksel region
    sy = 0
    sx = 0
    for (y, x) in piksel:
        sy += y
        sx += x
    return sy / len(piksel), sx / len(piksel)


def jari_jari(titik_batas, cy, cx):
    # jari-jari lingkaran penutup = jarak Euclid De TERJAUH dari pusat
    # ke titik batas:  De = sqrt((x - cx)^2 + (y - cy)^2)
    r = 0
    for (y, x) in titik_batas:
        d = math.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        if d > r:
            r = d
    return r


def kebundaran(luas, kel):
    # [DI LUAR SLIDE - Gonzalez] circularity = 4 pi A / P^2  (1 = lingkaran)
    if kel <= 0:
        return 0
    return 4 * math.pi * luas / (kel * kel)


def momen_region(piksel):
    # [DI LUAR SLIDE - Gonzalez] Momen pusat orde 2 -> elips yang setara.
    #   mu20, mu02, mu11 -> lambda1, lambda2 (sumbu panjang & pendek)
    #   lonjong = sqrt(lambda1 / lambda2)
    #   isi     = luas / luas elips (4 pi sqrt(l1 l2))  ~1 untuk bentuk bulat/lonjong
    #             rapi, kecil untuk bentuk aneh (huruf L, U, bercabang)
    n = len(piksel)
    cy, cx = pusat_region(piksel)
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


def signature(titik_batas, cy, cx):
    # Signature = jarak De dari pusat ke setiap titik batas, berurutan
    # mengikuti chain code. Lingkaran -> garis datar; segitiga -> 3 puncak;
    # persegi -> 4 puncak (puncak = sudut). Dihaluskan dengan rata-rata
    # bergeser (lowpass 1D) supaya tangga piksel tidak dianggap sudut.
    jarak = []
    for (y, x) in titik_batas:
        jarak.append(math.sqrt((y - cy) ** 2 + (x - cx) ** 2))
    n = len(jarak)
    w = n // 16
    if w < 2:
        w = 2
    halus = []
    for i in range(n):
        total = 0
        for k in range(-w, w + 1):
            total += jarak[(i + k) % n]
        halus.append(total / (2 * w + 1))
    return halus


def hitung_sudut(sig):
    # sudut = puncak signature: titik yang paling besar di jendela n/8
    # di kiri-kanannya dan berada di atas rata-rata
    n = len(sig)
    rata = sum(sig) / n
    w = n // 8
    if w < 2:
        w = 2
    sudut = 0
    for i in range(n):
        puncak = sig[i] > rata
        for k in range(-w, w + 1):
            if k != 0 and sig[(i + k) % n] >= sig[i]:
                # titik datar yang sama tinggi: hanya dihitung yang pertama
                if sig[(i + k) % n] > sig[i] or k < 0:
                    puncak = False
        if puncak:
            sudut += 1
    return sudut


def jenis_bentuk(titik_batas, cy, cx, lonjong, dua_warna):
    # Nama bentuk obat dari deskriptor region (materi 09):
    #   lonjong >= 1.35          -> kapsul (2 warna) atau oval / kaplet
    #   signature hampir datar   -> bulat
    #   jumlah sudut 3/4/5/6+    -> segitiga / persegi / segi lima / segi banyak
    if len(titik_batas) < 12:
        return "lainnya"
    if lonjong >= 1.35:
        if dua_warna:
            return "kapsul"
        return "oval / kaplet"
    sig = signature(titik_batas, cy, cx)
    rata = sum(sig) / len(sig)
    variasi = (max(sig) - min(sig)) / rata
    if variasi < 0.2:
        return "bulat"
    sudut = hitung_sudut(sig)
    if sudut == 3:
        return "segitiga"
    if sudut == 4:
        return "persegi / belah ketupat"
    if sudut == 5:
        return "segi lima"
    if sudut >= 6:
        return "segi banyak"
    # tidak ada sudut yang jelas -> tepi melengkung (bulat dengan tepi sedikit bergerigi)
    return "bulat"
