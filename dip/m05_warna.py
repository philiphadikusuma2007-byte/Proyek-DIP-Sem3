# =============================================================================
# dip/m05_warna.py  -  Materi 05 Color Image Processing
#
#   * Grayscale = komponen Y (luminance) dari model YIQ (slide 05):
#         Y = 0.299 R + 0.587 G + 0.114 B
#     ("Lihat hubungan Luminance dengan graylevel!" -> Y = graylevel)
#   * RGB -> HSI (slide 05 "RGB ke HSV", Gonzalez p.229-235):
#         I = (R + G + B) / 3
#         S = 1 - [3 / (R + G + B)] * min(R, G, B)
#         theta = cos^-1 { 1/2 [(R-G) + (R-B)] / [(R-G)^2 + (R-B)(G-B)]^(1/2) }
#         H = theta            bila B <= G
#         H = 360 - theta      bila B > G
#   * Jarak warna Euclid (De dari materi 02 dipakai di ruang RGB):
#         De = sqrt( (R1-R2)^2 + (G1-G2)^2 + (B1-B2)^2 )
#
# File ini TIDAK meng-import library apa pun (hanya modul dip sendiri).
# =============================================================================
from dip.dasar import PI, akar, arccos

# Tabel cos^-1 (derajat) untuk x = -1.000 .. 1.000 (langkah 0.001).
# Menghitung arccos dengan deret untuk 200 ribu piksel lambat, jadi
# hasilnya disimpan dulu di tabel (prinsip LUT yang sama dengan materi 02).
_TABEL_ACOS = [arccos(i / 1000.0 - 1.0) * 180.0 / PI for i in range(2001)]


def ke_abu(R, G, B):
    # graylevel = Y dari YIQ
    return [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in zip(R, G, B)]


def rgb_ke_hsi(R, G, B):
    # Hasil: H (derajat 0..360), S (0..1), I (0..255) untuk setiap piksel
    H = []
    S = []
    I = []
    tabel = _TABEL_ACOS
    for r, g, b in zip(R, G, B):
        total = r + g + b
        I.append(total / 3.0)
        mini = r
        if g < mini:
            mini = g
        if b < mini:
            mini = b
        if total == 0:
            S.append(0.0)
            H.append(0.0)
            continue
        S.append(1.0 - 3.0 * mini / total)
        atas = 0.5 * ((r - g) + (r - b))
        bawah = ((r - g) * (r - g) + (r - b) * (g - b)) ** 0.5
        if bawah == 0:
            H.append(0.0)                      # abu-abu murni: hue tidak terdefinisi
            continue
        x = atas / bawah                       # = cos(theta)
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        theta = tabel[int((x + 1.0) * 1000.0 + 0.5)]
        H.append(theta if b <= g else 360.0 - theta)
    return H, S, I


def hsi_satu_warna(warna):
    # rumus HSI yang sama, untuk satu warna [R, G, B]
    H, S, I = rgb_ke_hsi([warna[0]], [warna[1]], [warna[2]])
    return H[0], S[0], I[0]


def jarak_warna(R, G, B, warna):
    # jarak Euclid setiap piksel ke satu warna acuan [r, g, b]
    r0, g0, b0 = warna
    return [((r - r0) * (r - r0) + (g - g0) * (g - g0) + (b - b0) * (b - b0)) ** 0.5
            for r, g, b in zip(R, G, B)]


def jarak_dua_warna(w1, w2):
    return akar((w1[0] - w2[0]) ** 2 + (w1[1] - w2[1]) ** 2 + (w1[2] - w2[2]) ** 2)


def median_warna(R, G, B, indeks):
    # median R, G, B dari sekumpulan piksel (daftar indeks)
    if len(indeks) == 0:
        return [0, 0, 0]
    tengah = len(indeks) // 2
    return [sorted([K[i] for i in indeks])[tengah] for K in (R, G, B)]


def beda_hue(h1, h2):
    # hue berupa sudut: beda 350 dan 10 derajat = 20 derajat
    d = h1 - h2
    if d < 0:
        d = -d
    if d > 180:
        d = 360 - d
    return d
