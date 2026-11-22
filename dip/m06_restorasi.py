# =============================================================================
# m06_restorasi.py  -  Materi 06 Image Restoration
#   Model degradasi: g = f * h + eta  (eta = noise)
#   - Salt & pepper (impulse): banyak piksel ekstrem (0 / 255) yang beda
#     jauh dari tetangganya  -> order-statistic filter: MEDIAN
#   - Noise aditif (Gaussian): diperkirakan dari PATCH DATAR (blok yang
#     paling rata), mu = sum z p(z), sigma          -> mean filter / lowpass
#   - Noise periodik: titik terang di spektrum DFT  -> Butterworth NOTCH reject
#       H = 1 / (1 + [D0^2 / (D1 D2)]^n)
#       D1,2 = jarak ke titik noise (u0,v0) dan pasangan simetrisnya
# =============================================================================
import numpy as np

from dip.dasar import koordinat
from dip.m03_spasial import filter_median3, pad_tepi
from dip.m04_frekuensi import dft2, geser_tengah, spektrum


def cek_salt_pepper(abu):
    # persentase piksel ekstrem (<= 8 atau >= 247) yang beda > 50 dari median 3x3
    median = filter_median3(abu)
    ekstrem = (abu <= 8) | (abu >= 247)
    beda = (abu - median) ** 2 > 50 * 50
    jumlah = (ekstrem & beda).sum()
    return jumlah / abu.size


def estimasi_noise(abu):
    # Noise diukur di PATCH DATAR: residu = piksel - rata-rata 4 tetangga (N4).
    # Gambar dibagi blok 8x8; blok yang residunya kecil = daerah datar
    # (tidak ada tepi). Dipakai blok pada urutan 25% (bukan tepi, bukan
    # blok jenuh). Hasil = rata-rata |residu| di blok tersebut.
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


def nilai_tengah(daftar):
    # median sebuah daftar angka (diurutkan, ambil yang di tengah)
    urut = sorted(daftar)
    return urut[len(urut) // 2]


def cek_periodik(abu, radius_pusat, batas_puncak):
    # DFT 2D -> cari titik terang di luar pusat spektrum.
    # rasio = |F| / median |F|. Titik dengan rasio > batas_puncak = noise periodik.
    tinggi = abu.shape[0]
    lebar = abu.shape[1]
    F = geser_tengah(dft2(abu))
    besar = spektrum(F)

    cy = tinggi // 2
    cx = lebar // 2
    yy, xx = koordinat(tinggi, lebar)
    pusat = ((yy - cy) ** 2 + (xx - cx) ** 2 < radius_pusat ** 2) * 1.0
    median = nilai_tengah(besar[pusat < 0.5].tolist())
    rasio = besar / (median + 1)
    rasio[pusat > 0.5] = 0              # frekuensi rendah = isi gambar, bukan noise
    rasio[:cy, :] = 0                   # cukup separuh (spektrum simetris)

    puncak = []
    tertinggi = 0
    for k in range(4):                  # maksimal 4 pasang titik noise
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
        if terbaik < batas_puncak:
            break
        puncak.append((py, px))
        rasio[py - 4:py + 5, px - 4:px + 5] = 0
    return F, besar, puncak, tertinggi


def butterworth_notch(tinggi, lebar, daftar_puncak, d0, n):
    # Butterworth notch reject (slide 06), dikalikan untuk setiap titik noise:
    #   H = 1 / (1 + [D0^2 / (D1 D2)]^n)
    yy, xx = koordinat(tinggi, lebar)
    cy = tinggi // 2
    cx = lebar // 2
    H = np.ones((tinggi, lebar))
    for (py, px) in daftar_puncak:
        d1 = ((yy - py) ** 2 + (xx - px) ** 2) ** 0.5
        d2 = ((yy - (2 * cy - py)) ** 2 + (xx - (2 * cx - px)) ** 2) ** 0.5
        hasil_kali = d1 * d2
        hasil_kali[hasil_kali < 0.001] = 0.001
        H = H * (1 / (1 + (d0 * d0 / hasil_kali) ** n))
    return H
