# =============================================================================
# main.py  -  program utama
#   python main.py                      -> buka jendela (GUI)
#   python main.py foto1.jpg foto2.jpg  -> hasil + penjelasan langkah di terminal
#
# Pembagian tugas:
#   masukan.py   : membaca file gambar (library Pillow, hanya input)
#   deteksi.py   : SELURUH pengolahan citra (Python murni, tanpa library)
#   dip/         : rumus-rumus DIP per materi (Python murni, tanpa library)
#   tampilan.py  : jendela / GUI (tkinter + matplotlib + Pillow, hanya tampilan)
# =============================================================================
import os
import sys
import time

from masukan import baca_gambar
from deteksi import analisis_data


def analisis(path):
    # baca file -> proses -> tambahkan nama file & waktu proses
    mulai = time.time()
    lebar, tinggi, data = baca_gambar(path)
    hasil = analisis_data(lebar, tinggi, data)
    hasil["nama"] = os.path.basename(path)
    hasil["waktu"] = time.time() - mulai
    return hasil


def jalankan_terminal(daftar_path):
    for path in daftar_path:
        hasil = analisis(path)
        print("=" * 78)
        print(hasil["nama"], "-> SISA OBAT:", hasil["jumlah"], "| waktu", round(hasil["waktu"], 2), "detik")
        for k, o in enumerate(hasil["obat"]):
            print("   obat #" + str(k + 1) + ":", o["bentuk"], "| luas", o["luas"], "px | pusat (" +
                  str(int(o["cx"])) + ", " + str(int(o["cy"])) + ")")
        print("-" * 78)
        for k, langkah in enumerate(hasil["langkah"]):
            status = "" if langkah["dipakai"] else "  (dilewati)"
            print(str(k + 1) + ". " + langkah["judul"] + "  [" + langkah["materi"] + "]" + status)
            print("   " + langkah["rumus"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        jalankan_terminal(sys.argv[1:])
    else:
        from tampilan import jalankan_jendela
        jalankan_jendela(analisis)
