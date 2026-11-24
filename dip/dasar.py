# =============================================================================
# dip/dasar.py  -  Materi 01 & 02 Introduction to DIP
#
#   * Citra digital = matriks f(x,y). Citra warna = 3 matriks: R, G, B
#     (masing-masing 8 bit -> nilai 0..255).
#   * Di program ini SATU kanal disimpan sebagai list Python 1 dimensi
#     ("flat"), baris demi baris:
#
#         indeks i = y * lebar + x        x = i % lebar,  y = i // lebar
#
#     Jadi piksel (x, y) ada di data[y * lebar + x]. Tetangga kanan = i + 1,
#     tetangga bawah = i + lebar (konsep tetangga N4 / N8, materi 02).
#   * SAMPLING (digitisasi koordinat, materi 01): foto HP 12 megapiksel
#     terlalu besar untuk dihitung dengan Python murni, jadi diambil 1 blok
#     f x f piksel menjadi 1 piksel.
#   * Fungsi matematika dasar (akar, arctan, arccos) ditulis sendiri karena
#     program pengolahan citra TIDAK BOLEH memakai library apa pun
#     (termasuk modul math).
#
# File ini TIDAK meng-import apa pun.
# =============================================================================

PI = 3.141592653589793


# ----------------------------------------------------------------------------
# MATEMATIKA DASAR (tanpa modul math)
# ----------------------------------------------------------------------------
def akar(x):
    # akar kuadrat = x pangkat 1/2. Operator ** adalah operator bawaan
    # bahasa Python (bukan library).
    if x <= 0:
        return 0.0
    return x ** 0.5


def arctan(z):
    # arctan(z) dengan DERET TAYLOR:
    #     arctan(z) = z - z^3/3 + z^5/5 - z^7/7 + ...
    # Deret ini hanya cepat konvergen untuk |z| kecil, jadi z diperkecil dulu:
    #   (1) bila |z| > 1 :  arctan(z) = PI/2 - arctan(1/z)
    #   (2) rumus setengah sudut (dipakai 2x):
    #         arctan(z) = 2 * arctan( z / (1 + sqrt(1 + z^2)) )
    #       setelah 2x, |z| <= tan(PI/16) = 0.199, cukup 15 suku deret.
    tanda = 1.0
    if z < 0:
        tanda = -1.0
        z = -z
    dibalik = False
    if z > 1.0:
        z = 1.0 / z
        dibalik = True
    z = z / (1.0 + (1.0 + z * z) ** 0.5)       # setengah sudut ke-1
    z = z / (1.0 + (1.0 + z * z) ** 0.5)       # setengah sudut ke-2
    hasil = 0.0
    suku = z                                   # z^1, -z^3, z^5, ...
    z2 = z * z
    n = 1
    while n < 30:
        hasil += suku / n
        suku = -suku * z2
        n += 2
    hasil = hasil * 4.0                        # dikali 2 dua kali
    if dibalik:
        hasil = PI / 2 - hasil
    return tanda * hasil


def arctan2(y, x):
    # sudut vektor (x, y) dalam radian, rentang -PI .. PI
    if x > 0:
        return arctan(y / x)
    if x < 0:
        if y >= 0:
            return arctan(y / x) + PI
        return arctan(y / x) - PI
    if y > 0:
        return PI / 2
    if y < 0:
        return -PI / 2
    return 0.0


def arccos(x):
    # cos^-1(x) = arctan2( sqrt(1 - x^2), x )   untuk -1 <= x <= 1
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return arctan2(akar(1.0 - x * x), x)


def nilai_tengah(daftar):
    # median: data diurutkan, ambil yang di tengah
    if len(daftar) == 0:
        return 0
    urut = sorted(daftar)
    return urut[len(urut) // 2]


# ----------------------------------------------------------------------------
# SAMPLING (Materi 01: image sampling = digitisasi koordinat spasial)
# ----------------------------------------------------------------------------
def faktor_sampling(lebar, tinggi, maks_sisi):
    # f = berapa piksel asli digabung menjadi 1 piksel, supaya sisi
    # terpanjang hasil <= maks_sisi. Dibulatkan ke atas.
    sisi = lebar
    if tinggi > sisi:
        sisi = tinggi
    f = (sisi + maks_sisi - 1) // maks_sisi
    if f < 1:
        f = 1
    return f


def sampling(lebar, tinggi, data, maks_sisi):
    # sampling seluruh foto
    return sampling_kotak(lebar, tinggi, data, 0, 0, lebar, tinggi, maks_sisi)


def sampling_kotak(lebar, tinggi, data, kx0, ky0, kx1, ky1, maks_sisi, f=0):
    # data = bytes RGB mentah dari file gambar: R,G,B,R,G,B,... baris demi baris
    # (indeks byte piksel (x, y) kanal c = (y * lebar + x) * 3 + c).
    # Yang diambil hanya kotak [kx0, kx1) x [ky0, ky1) (cropping, materi 01).
    #
    # Setiap blok f x f piksel diwakili 4 titik sampel (posisi 1/4 dan 3/4
    # blok di arah x dan y) yang DIRATA-RATA:
    #       g(m, n) = [ f(a,a) + f(a,b) + f(b,a) + f(b,b) ] / 4
    # Rata-rata ini = lowpass kecil (materi 03) sebelum sampling, sehingga
    # noise berkurang dan pola halus foil tidak menimbulkan aliasing.
    # Hasil: lebar baru, tinggi baru, list R, list G, list B (int 0..255), f.
    # f bisa ditentukan langsung (f > 0); bila 0 dihitung dari maks_sisi.
    if f <= 0:
        f = faktor_sampling(kx1 - kx0, ky1 - ky0, maks_sisi)
    lebar2 = (kx1 - kx0) // f
    tinggi2 = (ky1 - ky0) // f
    a = f // 4                    # posisi sampel pertama di dalam blok
    b = (3 * f) // 4              # posisi sampel kedua di dalam blok
    langkah = 3 * f               # loncat f piksel = 3f byte
    R = []
    G = []
    B = []
    for j in range(tinggi2):
        y1 = ky0 + j * f + a
        y2 = ky0 + j * f + b
        for c, kanal in ((0, R), (1, G), (2, B)):
            # slicing bytes [awal : akhir : langkah] mengambil kanal c dari
            # setiap piksel ke-f pada satu baris (operasi bawaan Python)
            awal = (y1 * lebar + kx0 + a) * 3 + c
            s1 = data[awal:awal + langkah * lebar2:langkah]
            awal = (y1 * lebar + kx0 + b) * 3 + c
            s2 = data[awal:awal + langkah * lebar2:langkah]
            awal = (y2 * lebar + kx0 + a) * 3 + c
            s3 = data[awal:awal + langkah * lebar2:langkah]
            awal = (y2 * lebar + kx0 + b) * 3 + c
            s4 = data[awal:awal + langkah * lebar2:langkah]
            # rata-rata 4 sampel, dibulatkan: (jumlah + 2) // 4
            kanal.extend([(p + q + r + s + 2) >> 2 for p, q, r, s in zip(s1, s2, s3, s4)])
    return lebar2, tinggi2, R, G, B, f


def potong(data, lebar, x0, y0, x1, y1):
    # cropping: ambil kotak [x0, x1) x [y0, y1) dari satu kanal
    hasil = []
    for y in range(y0, y1):
        hasil.extend(data[y * lebar + x0:y * lebar + x1])
    return hasil


def jepit(data, bawah, atas):
    # clipping: nilai < bawah -> bawah, nilai > atas -> atas
    return [bawah if v < bawah else (atas if v > atas else v) for v in data]


def ke_int(data):
    # bulatkan setiap nilai ke bilangan bulat terdekat lalu jepit 0..255
    return [0 if v < 0 else (255 if v > 255 else int(v + 0.5)) for v in data]


def transpos(data, lebar, tinggi):
    # tukar baris <-> kolom (piksel (x, y) pindah ke (y, x)).
    # data[x::lebar] = seluruh kolom ke-x (slicing bawaan Python).
    hasil = []
    for x in range(lebar):
        hasil.extend(data[x::lebar])
    return hasil
