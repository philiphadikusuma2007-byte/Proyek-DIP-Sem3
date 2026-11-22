# =============================================================================
# main.py  -  program utama
#   python main.py                      -> buka jendela (GUI)
#   python main.py foto1.jpg foto2.jpg  -> hasil di terminal saja
# =============================================================================
import sys

from deteksi import analisis


def jalankan_terminal(daftar_path):
    for path in daftar_path:
        hasil = analisis(path)
        print(hasil["nama"], "-> sisa obat:", hasil["jumlah"],
              "| waktu", round(hasil["waktu"], 2), "detik")
        bentuk = []
        for o in hasil["obat"]:
            bentuk.append(o["bentuk"] + (" x" + str(o["butir"]) if o["butir"] > 1 else ""))
        if len(bentuk) > 0:
            print("    bentuk obat:", ", ".join(bentuk))
        nomor = 0
        for langkah in hasil["langkah"]:
            nomor += 1
            if langkah["dipakai"]:
                print("   ", str(nomor) + ".", langkah["judul"], "[" + langkah["materi"] + "]")
        alasan = []
        for d in hasil["ditolak"]:
            alasan.append(d["alasan"])
        if len(alasan) > 0:
            print("    ditolak:", ", ".join(alasan))


if len(sys.argv) > 1:
    jalankan_terminal(sys.argv[1:])
else:
    from tampilan import jalankan_jendela
    jalankan_jendela(analisis)
