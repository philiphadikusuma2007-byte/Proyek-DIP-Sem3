# =============================================================================
# m04_frekuensi.py  -  Materi 04 Frequency Domain
#   - DFT 2D:  F(u,v) = sum_x sum_y f(x,y) e^{-j 2 pi (ux/M + vy/N)}
#     dihitung separabel: DFT 1D tiap baris, lalu tiap kolom
#   - spektrum |F| = sqrt(R^2 + I^2), digeser supaya F(0,0) di tengah
#   - Butterworth HPF: H = 1 / (1 + [r0 / D]^{2n})
# =============================================================================
import math
import numpy as np


def dft_tiap_baris(a, tanda):
    # DFT 1D pada setiap baris:  F(u) = sum_x f(x) e^{tanda j 2 pi u x / N}
    # tanda = -1 untuk DFT, +1 untuk invers.
    # Semua baris dihitung bersamaan (yang di-loop hanya u dan x).
    tinggi = a.shape[0]
    n = a.shape[1]
    kolom = a.T.copy()                   # kolom[x] = nilai kolom ke-x (semua baris)
    hasil = np.zeros((n, tinggi), dtype=complex)
    for u in range(n):
        jumlah = np.zeros(tinggi, dtype=complex)
        for x in range(n):
            sudut = 2 * math.pi * ((u * x) % n) / n
            jumlah += kolom[x] * complex(math.cos(sudut), tanda * math.sin(sudut))
        hasil[u] = jumlah
    return hasil.T


def dft2(a):
    # DFT 2D separabel: per baris, lalu per kolom
    F = dft_tiap_baris(a, -1)
    F = dft_tiap_baris(F.T, -1).T
    return F


def idft2(F):
    # invers DFT 2D, dibagi M*N
    tinggi = F.shape[0]
    lebar = F.shape[1]
    f = dft_tiap_baris(F, 1)
    f = dft_tiap_baris(f.T, 1).T
    return f / (tinggi * lebar)


def geser_tengah(F):
    # pindahkan frekuensi (0,0) ke tengah spektrum (tukar 4 kuadran)
    tinggi = F.shape[0]
    lebar = F.shape[1]
    hasil = np.zeros(F.shape, dtype=F.dtype)
    for y in range(tinggi):
        hasil[(y + tinggi // 2) % tinggi] = np.concatenate(
            (F[y, lebar - lebar // 2:], F[y, :lebar - lebar // 2]))
    return hasil


def spektrum(F):
    # |F(u,v)| = sqrt(R^2 + I^2)
    return (F.real ** 2 + F.imag ** 2) ** 0.5
