# =============================================================================
# dip/m08_segmentasi.py  -  Materi 08 Image Segmentation (+ konektivitas 02)
#
#   * Thresholding: piksel > T -> objek (single thresholding, slide 08).
#     T dipilih dari histogram. Cara memilih T otomatis = metode OTSU
#     (buku Gonzalez, DI LUAR slide): murni rumus statistik histogram,
#     tanpa data latih / tanpa "belajar" -> bukan machine learning.
#   * Pelabelan region (connected component): piksel objek yang saling
#     terhubung N4 atau N8 (slide 02 "Connectivity") diberi nomor yang sama.
#   * Region growing (slide 08): mulai dari seed, region membesar dengan
#     menambahkan piksel tetangga yang sifatnya hampir sama.
#
# File ini TIDAK meng-import library apa pun.
# =============================================================================


def otsu(hist):
    # [Gonzalez - Otsu] Pilih T (0..255) yang membuat dua kelas
    #   kelas 1: piksel <= T     kelas 2: piksel > T
    # paling terpisah, yaitu variansi ANTAR-kelas terbesar:
    #   sigma_B^2(T) = w1(T) * w2(T) * (mu1(T) - mu2(T))^2
    # w = proporsi piksel tiap kelas, mu = rata-rata tiap kelas.
    n = 0
    total = 0
    for k in range(256):
        n += hist[k]
        total += k * hist[k]
    if n == 0:
        return 128
    terbaik = -1.0
    batas = 0
    n1 = 0
    total1 = 0
    for t in range(256):
        n1 += hist[t]
        total1 += t * hist[t]
        n2 = n - n1
        if n1 == 0:
            continue
        if n2 == 0:
            break
        mu1 = total1 / n1
        mu2 = (total - total1) / n2
        sb = (n1 / n) * (n2 / n) * (mu1 - mu2) * (mu1 - mu2)
        if sb > terbaik:
            terbaik = sb
            batas = t
    return batas


def label_region(mask, lebar, tinggi, konektivitas=8):
    # Pelabelan region: setiap kelompok piksel bernilai 1 yang tersambung
    # (N4 atau N8) diberi nomor 1, 2, 3, ...  Memakai tumpukan (stack) =
    # flood fill. Gambar diberi bingkai 0 selebar 1 piksel supaya tetangga
    # di tepi tidak perlu dicek batasnya.
    L = lebar + 2
    pad = [0] * (L * (tinggi + 2))
    for y in range(tinggi):
        pad[(y + 1) * L + 1:(y + 1) * L + 1 + lebar] = mask[y * lebar:(y + 1) * lebar]
    if konektivitas == 4:
        arah = (-L, L, -1, 1)                                   # N4(p)
    else:
        arah = (-L, L, -1, 1, -L - 1, -L + 1, L - 1, L + 1)     # N8(p) = N4 + ND
    label = [0] * len(pad)
    daftar = []
    nomor = 0
    for awal in range(len(pad)):
        if pad[awal] == 0 or label[awal] != 0:
            continue
        nomor += 1
        label[awal] = nomor
        tumpukan = [awal]
        piksel = []
        while tumpukan:
            p = tumpukan.pop()
            piksel.append(p)
            for d in arah:
                q = p + d
                if pad[q] and not label[q]:
                    label[q] = nomor
                    tumpukan.append(q)
        # indeks berbingkai -> indeks gambar asli, sekaligus kotak pembatas
        asli = []
        x0 = lebar
        y0 = tinggi
        x1 = -1
        y1 = -1
        for p in piksel:
            y = p // L - 1
            x = p % L - 1
            asli.append(y * lebar + x)
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
        daftar.append({"nomor": nomor, "luas": len(asli), "piksel": asli,
                       "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    peta = []
    for y in range(tinggi):
        peta.extend(label[(y + 1) * L + 1:(y + 1) * L + 1 + lebar])
    return peta, daftar


def mask_dari_region(daftar_region, lebar, tinggi):
    # gambar biner yang berisi region-region terpilih
    m = [0] * (lebar * tinggi)
    for d in daftar_region:
        for i in d["piksel"]:
            m[i] = 1
    return m


def region_growing(seed, boleh, lebar, tinggi):
    # Region growing (slide 08): region mulai dari piksel seed (mask 0/1),
    # lalu berulang kali menambah tetangga N8 yang memenuhi syarat
    # kemiripan (mask "boleh" = 1). Berhenti bila tidak ada lagi tetangga
    # yang memenuhi syarat.
    hasil = list(seed)
    tumpukan = [i for i in range(len(seed)) if seed[i]]
    while tumpukan:
        p = tumpukan.pop()
        x = p % lebar
        y = p // lebar
        for dy in (-1, 0, 1):
            ny = y + dy
            if ny < 0 or ny >= tinggi:
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                if nx < 0 or nx >= lebar:
                    continue
                q = ny * lebar + nx
                if boleh[q] and not hasil[q]:
                    hasil[q] = 1
                    tumpukan.append(q)
    return hasil
