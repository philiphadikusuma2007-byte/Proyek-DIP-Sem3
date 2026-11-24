# =============================================================================
# dip/m09_representasi.py  -  Materi 09 Representation & Description
#
#   MORFOLOGI biner (slide 09 + visualisasi_morfologi_interaktif.html):
#     - Dilasi : output 1 bila MINIMAL SATU posisi aktif SE mengenai piksel 1
#     - Erosi  : output 1 bila SEMUA posisi aktif SE mengenai piksel 1
#     - Opening = erosi lalu dilasi  -> memisahkan objek yang salah menyatu,
#                                       membuang bintik kecil
#     - Closing = dilasi lalu erosi  -> menyatukan objek yang salah terpisah,
#                                       menutup celah kecil
#     SE yang dipakai = "semua arah 2" (kotak penuh) diperbesar menjadi k x k.
#     Untuk kotak k x k:  erosi  -> jumlah 1 di jendela == k*k
#                         dilasi -> jumlah 1 di jendela  > 0
#     (jumlah jendela dihitung dengan filter rata-rata/jumlah materi 03).
#
#   MORFOLOGI graylevel (perluasan, Gonzalez): erosi = filter MINIMUM,
#     dilasi = filter MAKSIMUM (keduanya filter non-linear slide 03).
#     Top-hat = f - opening(f)  -> menonjolkan objek TERANG yang lebih kecil
#     dari SE (di sini: obat yang lebih terang dari foil di sekitarnya).
#
#   REPRESENTASI & DESKRIPSI:
#     - Kode rantai (chain code) 8 arah:   3 2 1
#                                          4 x 0
#                                          5 6 7
#     - keliling dari chain code: kode genap = 1, kode ganjil = sqrt(2)
#     - momen region (Gonzalez, di luar slide): pusat, elips setara,
#       kelonjongan, keterisian
#
# File ini TIDAK meng-import library apa pun (hanya modul dip sendiri).
# =============================================================================
from dip.dasar import PI, akar
from dip.m03_spasial import jumlah_kotak, filter_minimum, filter_maksimum


# ----------------------------------------------------------------------------
# MORFOLOGI BINER (SE kotak k x k)
# ----------------------------------------------------------------------------
def erosi(mask, lebar, tinggi, k):
    if k <= 1:
        return list(mask)
    penuh = k * k
    return [1 if s == penuh else 0 for s in jumlah_kotak(mask, lebar, tinggi, k)]


def dilasi(mask, lebar, tinggi, k):
    if k <= 1:
        return list(mask)
    return [1 if s > 0 else 0 for s in jumlah_kotak(mask, lebar, tinggi, k)]


def opening(mask, lebar, tinggi, k):
    return dilasi(erosi(mask, lebar, tinggi, k), lebar, tinggi, k)


def closing(mask, lebar, tinggi, k):
    return erosi(dilasi(mask, lebar, tinggi, k), lebar, tinggi, k)


def isi_lubang(mask, lebar, tinggi, maks_luas=0):
    # [Gonzalez - hole filling] Lubang = daerah 0 yang TIDAK tersambung ke
    # tepi gambar. Caranya: telusuri (flood fill N4) semua piksel 0 mulai dari
    # tepi gambar; piksel 0 yang tidak tercapai adalah lubang -> dijadikan 1.
    # Bila maks_luas > 0, hanya lubang yang luasnya <= maks_luas yang diisi
    # (mis. huruf timbul di permukaan obat), lubang besar dibiarkan.
    luar = [0] * (lebar * tinggi)
    tumpukan = []
    for x in range(lebar):
        for i in (x, (tinggi - 1) * lebar + x):
            if mask[i] == 0 and not luar[i]:
                luar[i] = 1
                tumpukan.append(i)
    for y in range(tinggi):
        for i in (y * lebar, y * lebar + lebar - 1):
            if mask[i] == 0 and not luar[i]:
                luar[i] = 1
                tumpukan.append(i)
    while tumpukan:
        p = tumpukan.pop()
        x = p % lebar
        for q, ok in ((p - lebar, p >= lebar), (p + lebar, p < (tinggi - 1) * lebar),
                      (p - 1, x > 0), (p + 1, x < lebar - 1)):
            if ok and mask[q] == 0 and not luar[q]:
                luar[q] = 1
                tumpukan.append(q)
    if maks_luas <= 0:
        return [0 if l else 1 for l in luar]
    # hanya lubang kecil: label setiap lubang, isi yang kecil saja
    hasil = list(mask)
    lubang = [1 if (not l and not m) else 0 for l, m in zip(luar, mask)]
    tanda = [0] * (lebar * tinggi)
    for awal in range(lebar * tinggi):
        if not lubang[awal] or tanda[awal]:
            continue
        tanda[awal] = 1
        tumpukan = [awal]
        anggota = []
        while tumpukan:
            p = tumpukan.pop()
            anggota.append(p)
            x = p % lebar
            for q, ok in ((p - lebar, p >= lebar), (p + lebar, p < (tinggi - 1) * lebar),
                          (p - 1, x > 0), (p + 1, x < lebar - 1)):
                if ok and lubang[q] and not tanda[q]:
                    tanda[q] = 1
                    tumpukan.append(q)
        if len(anggota) <= maks_luas:
            for p in anggota:
                hasil[p] = 1
    return hasil


def batas(mask, lebar, tinggi):
    # garis batas = A - erosi(A)  (dibangun dari erosi slide 09)
    e = erosi(mask, lebar, tinggi, 3)
    return [a - b for a, b in zip(mask, e)]


# ----------------------------------------------------------------------------
# MORFOLOGI GRAYLEVEL
# ----------------------------------------------------------------------------
def opening_abu(data, lebar, tinggi, k):
    # opening graylevel = filter minimum lalu filter maksimum (SE k x k)
    return filter_maksimum(filter_minimum(data, lebar, tinggi, k), lebar, tinggi, k)


def top_hat(data, lebar, tinggi, k):
    # top-hat putih = f - opening(f).  Objek terang yang LEBIH KECIL dari SE
    # hilang oleh opening, jadi selisihnya = objek terang tersebut.
    o = opening_abu(data, lebar, tinggi, k)
    return [a - b for a, b in zip(data, o)], o


# ----------------------------------------------------------------------------
# KODE RANTAI & DESKRIPSI REGION
# ----------------------------------------------------------------------------
# arah chain code 8 (dx, dy) dengan sumbu y ke BAWAH (koordinat gambar):
#   0 kanan, 1 kanan-atas, 2 atas, 3 kiri-atas, 4 kiri, 5 kiri-bawah,
#   6 bawah, 7 kanan-bawah
ARAH_8 = ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))


def chain_code(piksel, lebar):
    # Menelusuri batas LUAR region (Moore boundary tracing) mulai dari piksel
    # paling atas-kiri, berlawanan arah jarum jam. Setiap langkah ke
    # tetangga N8 dicatat kodenya (0..7). Hasil: daftar kode + titik batas.
    isi = set(piksel)
    mulai = min(piksel)                       # indeks terkecil = paling atas-kiri
    sx = mulai % lebar
    sy = mulai // lebar
    kode = []
    titik = [(sx, sy)]
    x = sx
    y = sy
    arah = 7                                  # anggap datang dari kiri-bawah
    batas_langkah = 4 * len(piksel) + 8
    while len(kode) < batas_langkah:
        # mulai mencari dari arah "belakang-kiri" arah terakhir
        cari = (arah + 7) % 8 if arah % 2 == 0 else (arah + 6) % 8
        ketemu = False
        for j in range(8):
            d = (cari + j) % 8
            nx = x + ARAH_8[d][0]
            ny = y + ARAH_8[d][1]
            if 0 <= nx < lebar and ny >= 0 and ny * lebar + nx in isi:
                kode.append(d)
                x = nx
                y = ny
                titik.append((x, y))
                arah = d
                ketemu = True
                break
        if not ketemu:                        # region hanya 1 piksel
            break
        if x == sx and y == sy:
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
    return genap + ganjil * akar(2.0)


def momen(piksel, lebar):
    # [Gonzalez - momen region] Hasil: pusat (cx, cy), lambda1 >= lambda2
    # (variansi sepanjang sumbu panjang & pendek), kelonjongan, keterisian.
    #   pusat  cx = rata-rata x,  cy = rata-rata y
    #   mu20 = rata (x-cx)^2, mu02 = rata (y-cy)^2, mu11 = rata (x-cx)(y-cy)
    #   lambda1,2 = (mu20+mu02)/2 +- sqrt( ((mu20-mu02)/2)^2 + mu11^2 )
    #   elips setara: setengah sumbu a = 2 sqrt(lambda1), b = 2 sqrt(lambda2)
    #   kelonjongan = a / b      (1 = bulat, 2.5 = kaplet / kapsul)
    #   keterisian  = luas / luas elips (pi a b)  (~1 untuk bentuk bulat /
    #                 lonjong yang rapi, kecil untuk bentuk aneh: L, U, cincin)
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
    beda = akar(((m20 - m02) / 2) ** 2 + m11 * m11)
    l1 = tengah + beda
    l2 = tengah - beda
    if l2 < 0.25:
        l2 = 0.25
    a = 2 * akar(l1)
    b = 2 * akar(l2)
    return {"cx": cx, "cy": cy, "a": a, "b": b, "lonjong": a / b,
            "isi": n / (PI * a * b), "m20": m20, "m02": m02, "m11": m11}
