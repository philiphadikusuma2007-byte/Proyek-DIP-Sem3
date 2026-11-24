# =============================================================================
# masukan.py  -  INPUT GAMBAR (satu-satunya bagian yang memakai library
#                untuk membaca file)
#
# File JPG/PNG tersimpan dalam format terkompresi (JPEG = DCT + Huffman,
# materi 07). Membuka (decode) format file itu adalah tugas "input gambar",
# jadi di sini dipakai library Pillow. Hasilnya diserahkan ke program utama
# dalam bentuk paling mentah: lebar, tinggi, dan deretan byte R,G,B,R,G,B,...
#
# SEMUA pengolahan citra (sampling, grayscale, filter, segmentasi, ...)
# dikerjakan di deteksi.py + folder dip/ dengan Python murni tanpa library.
# =============================================================================
from PIL import Image, ImageOps


def baca_gambar(path):
    img = Image.open(path)
    # foto HP menyimpan arah putar di metadata EXIF; diterapkan supaya foto
    # tidak terbaca miring/terbalik
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")                 # PNG RGBA / grayscale -> 3 kanal
    lebar, tinggi = img.size
    return lebar, tinggi, img.tobytes()      # bytes: R,G,B,R,G,B,... per baris
