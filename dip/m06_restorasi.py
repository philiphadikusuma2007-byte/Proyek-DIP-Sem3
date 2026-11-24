# =============================================================================
# dip/m06_restorasi.py  -  Materi 06 Image Restoration
#
#   Model degradasi (slide 06):  g(m,n) = f(m,n) * h(m,n) + eta(m,n)
#   Bila degradasi hanya berupa noise:
#     - Salt & pepper (impulse): piksel ekstrem (hitam / putih) yang beda
#       jauh dari tetangganya  -> order-statistic filter: MEDIAN
#     - Noise aditif (mis. Gaussian): diperkirakan di PATCH DATAR (daerah
#       tanpa tepi), lalu dibersihkan dengan mean filter / lowpass
#
#   Program MEMERIKSA dulu, baru memakai filter bila memang perlu, supaya
#   tepi obat tidak ikut kabur tanpa alasan.
#
# File ini TIDAK meng-import library apa pun (hanya modul dip sendiri).
# =============================================================================
from dip.m03_spasial import median3


def cek_salt_pepper(abu, lebar, tinggi):
    # Persentase piksel impulse: nilainya ekstrem (<= 10 atau >= 245) DAN
    # beda lebih dari 60 dari median 3x3 tetangganya.
    med = median3(abu, lebar, tinggi)
    n = 0
    for v, m in zip(abu, med):
        if (v <= 10 or v >= 245) and (v - m > 60 or m - v > 60):
            n += 1
    return n / float(len(abu)), med


def estimasi_noise(abu, lebar, tinggi):
    # Noise diukur di PATCH DATAR:
    #   residu = piksel - rata-rata 4 tetangga (N4)
    # Gambar dibagi blok 8x8. Blok yang residunya kecil = daerah datar (tidak
    # ada tepi). Dipakai blok urutan 25% terdatar supaya tepi obat / tekstur
    # foil tidak dianggap noise. Hasil = rata-rata |residu| blok tersebut.
    nilai_blok = []
    for by in range(1, tinggi // 8 - 1):
        for bx in range(1, lebar // 8 - 1):
            total = 0.0
            for y in range(by * 8, by * 8 + 8):
                i = y * lebar + bx * 8
                for x in range(8):
                    p = i + x
                    r = abu[p] - (abu[p - 1] + abu[p + 1] + abu[p - lebar] + abu[p + lebar]) / 4.0
                    total += r if r > 0 else -r
            nilai_blok.append(total / 64.0)
    if len(nilai_blok) == 0:
        return 0.0
    nilai_blok.sort()
    return nilai_blok[len(nilai_blok) // 4]
