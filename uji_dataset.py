# =============================================================================
# uji_dataset.py - menguji program pada semua foto di folder gambar/
# dan membandingkan dengan jawaban manual di gambar/jawaban.txt
# Cara pakai:  python uji_dataset.py          (ringkas)
#              python uji_dataset.py -v       (tampilkan detail setiap region)
# =============================================================================
import os
import sys

from deteksi import analisis

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gambar")
detail = "-v" in sys.argv
pilihan = [a for a in sys.argv[1:] if a != "-v"]

benar = 0
total = 0
with open(os.path.join(FOLDER, "jawaban.txt"), encoding="utf8") as f:
    for baris in f:
        baris = baris.strip()
        if baris == "" or baris.startswith("#"):
            continue
        nama, jawaban = baris.split("|")
        if pilihan and not any(p in nama for p in pilihan):
            continue
        hasil = analisis(os.path.join(FOLDER, nama), print if detail else None)
        total += 1
        cocok = hasil["jumlah"] == int(jawaban)
        benar += cocok
        print("%-50s jawaban %2s  program %2d  (potongan %d)  %5.1f s  %s" %
              (nama, jawaban, hasil["jumlah"], hasil["jumlah_potongan"], hasil["waktu"],
               "BENAR" if cocok else "SALAH"))
print("\nAkurasi: %d / %d" % (benar, total))
