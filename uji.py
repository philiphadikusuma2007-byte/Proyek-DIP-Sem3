# =============================================================================
# uji.py  -  UJI OTOMATIS terhadap kunci jawaban (gambar/jawaban.txt)
#   python uji.py            -> proses semua foto di folder gambar/
#   python uji.py --gambar   -> juga simpan gambar hasil ke folder hasil_uji/
#
# Ini alat bantu pengujian (bukan bagian program utama).
# =============================================================================
import os
import sys

from main import analisis

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gambar")


def baca_kunci():
    kunci = {}
    with open(os.path.join(FOLDER, "jawaban.txt"), encoding="utf-8") as berkas:
        for baris in berkas:
            baris = baris.strip()
            if baris and not baris.startswith("#") and "|" in baris:
                nama, jumlah = baris.split("|")
                kunci[nama.strip()] = int(jumlah)
    return kunci


def simpan_gambar(hasil, folder):
    from matplotlib.figure import Figure
    from tampilan import gambar_hasil
    fig = Figure(figsize=(6, 6), dpi=100)
    gambar_hasil(fig.add_subplot(1, 1, 1), hasil, True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    fig.suptitle(hasil["nama"] + "  -  sisa obat " + str(hasil["jumlah"]), fontsize=9)
    fig.savefig(os.path.join(folder, os.path.splitext(hasil["nama"])[0].replace(" ", "_") + ".png"))


def main():
    kunci = baca_kunci()
    folder_hasil = None
    if "--gambar" in sys.argv:
        folder_hasil = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_uji")
        os.makedirs(folder_hasil, exist_ok=True)
    benar = 0
    total_selisih = 0
    total_waktu = 0.0
    nama_file = sorted(f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for nama in nama_file:
        hasil = analisis(os.path.join(FOLDER, nama))
        harapan = kunci.get(nama)
        cocok = harapan is not None and hasil["jumlah"] == harapan
        benar += cocok
        if harapan is not None:
            total_selisih += abs(hasil["jumlah"] - harapan)
        total_waktu += hasil["waktu"]
        print("%-4s %-50s hasil %2d | kunci %2s | %.1f s" % ("OK" if cocok else "SALAH", nama, hasil["jumlah"],
                                                              harapan, hasil["waktu"]))
        if folder_hasil:
            simpan_gambar(hasil, folder_hasil)
    print("-" * 80)
    print("Benar %d dari %d foto, total selisih %d butir, rata-rata %.1f detik per foto" % (
        benar, len(nama_file), total_selisih, total_waktu / max(len(nama_file), 1)))


if __name__ == "__main__":
    main()
