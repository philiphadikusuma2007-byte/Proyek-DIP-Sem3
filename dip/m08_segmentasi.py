# =============================================================================
# m08_segmentasi.py  -  Materi 08 Image Segmentation
#   - Deteksi tepi Sobel:  grad f = |Gx| + |Gy|
#       Gx = -Z1 - 2Z2 - Z3 + Z7 + 2Z8 + Z9,  Gy = -Z1 - 2Z4 - Z7 + Z3 + 2Z6 + Z9
#   - Thresholding: piksel > T -> objek. T dipilih dari histogram.
#     Otsu = cara memilih T otomatis (Gonzalez, DI LUAR slide).
#   - Region growing: mulai dari seed, tambahkan tetangga yang mirip.
#   - Pelabelan region dengan konektivitas N4 / N8 (02 Intro II).
# =============================================================================
import numpy as np

from dip.m03_spasial import konvolusi, SOBEL_H1, SOBEL_H2, filter_maksimum


def sobel(abu):
    # besar gradien = |Gx| + |Gy|  (rumus slide 08)
    gx = konvolusi(abu, SOBEL_H1)
    gy = konvolusi(abu, SOBEL_H2)
    gx[gx < 0] = -gx[gx < 0]
    gy[gy < 0] = -gy[gy < 0]
    return gx + gy


def otsu(hist):
    # [DI LUAR SLIDE - Gonzalez] Otsu: pilih T yang membuat dua kelas
    # (<= T dan > T) paling terpisah = variansi antar-kelas terbesar:
    #   sigma_B^2(T) = n_bawah * n_atas * (rata_bawah - rata_atas)^2
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


def tetangga(konektivitas):
    # N4(p): atas, bawah, kiri, kanan.  N8(p) = N4(p) + 4 diagonal (ND)
    n4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if konektivitas == 4:
        return n4
    return n4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def label_region(mask, konektivitas=4):
    # Pelabelan region: setiap kelompok piksel 1 yang tersambung (N4 / N8)
    # diberi nomor. Memakai tumpukan (stack) = flood fill.
    # Hasil: peta label + daftar region (luas, kotak pembatas, daftar piksel).
    tinggi = mask.shape[0]
    lebar = mask.shape[1]
    arah = tetangga(konektivitas)
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
                    py, px = tumpukan.pop()
                    piksel.append((py, px))
                    if px < x0:
                        x0 = px
                    if px > x1:
                        x1 = px
                    if py < y0:
                        y0 = py
                    if py > y1:
                        y1 = py
                    for (dy, dx) in arah:
                        ny = py + dy
                        nx = px + dx
                        if ny >= 0 and ny < tinggi and nx >= 0 and nx < lebar:
                            if data[ny][nx] and label[ny][nx] == 0:
                                label[ny][nx] = nomor
                                tumpukan.append((ny, nx))
                daftar.append({"nomor": nomor, "luas": len(piksel), "piksel": piksel,
                               "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    return np.array(label), daftar


def region_terbesar(mask):
    # simpan hanya region dengan luas terbesar
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


def region_growing(seed, boleh, langkah):
    # Region growing: region mulai dari seed (mask 0/1), lalu berulang kali
    # menambah tetangga N8 yang MEMENUHI kriteria kemiripan (mask "boleh").
    # Setiap putaran = region melebar 1 piksel.
    region = seed.copy()
    for i in range(langkah):
        baru = filter_maksimum(region, 3) * boleh
        baru[seed > 0.5] = 1
        region = baru
    return region
