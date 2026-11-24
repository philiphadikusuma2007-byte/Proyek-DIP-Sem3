# Deteksi Sisa Obat dalam Blister — Penjelasan Program

Program menghitung **jumlah obat yang masih ada** di dalam kemasan blister dari sebuah foto,
dan menampilkan **setiap langkah pengolahan citra** (gambar + rumus + penjelasan dengan angka
asli dari foto tersebut).

## Aturan yang dipenuhi

| Aturan | Cara dipenuhi |
|---|---|
| Tanpa Machine Learning | Tidak ada model, data latih, klasifikasi terlatih, k-means, jaringan saraf, dll. Semua keputusan berupa **rumus dan aturan** dari materi DIP (threshold, morfologi, deskriptor region). |
| Program utama tanpa library | `deteksi.py` dan folder `dip/` **tidak meng-import apa pun** — bahkan `math` pun tidak. Yang di-import hanya modul buatan sendiri di `dip/`. |
| Library hanya untuk UI & input gambar | `masukan.py` (Pillow: membuka file JPG/PNG), `tampilan.py` (tkinter + matplotlib + Pillow: menampilkan). |
| Rumus manual | Grayscale, HSI, YIQ, histogram, Otsu, konvolusi, median, filter min/maks, morfologi, pelabelan, chain code, momen — semuanya ditulis sendiri. `akar`, `arctan`, `arccos` pun ditulis dengan rumus (deret Taylor). |

Bukti: `grep -nE "^\s*(import|from)\s" deteksi.py dip/*.py` hanya menampilkan `from dip...`.

## Cara menjalankan

```
python main.py                      # jendela GUI
python main.py gambar/foto.jpg      # hasil + daftar langkah di terminal
python uji.py                       # uji semua foto terhadap gambar/jawaban.txt
python uji.py --gambar              # + simpan gambar hasil ke folder hasil_uji/
```

Di GUI: **Pilih Gambar** → **Deteksi** → klik langkah di kiri (atau tombol ← / →). Halaman
"Hasil akhir" menampilkan elips hijau di setiap obat; centang *Tampilkan region yang ditolak*
untuk melihat region yang ditolak beserta alasannya. **Simpan Hasil** menyimpan gambar hasil +
seluruh penjelasan langkah (teks) ke folder `hasil/`.

## Struktur file

| File | Isi |
|---|---|
| `main.py` | Program utama: GUI atau mode terminal. Mengukur waktu proses. |
| `masukan.py` | **Input gambar** (Pillow): membuka file, memutar sesuai EXIF, mengeluarkan bytes R,G,B mentah. |
| `deteksi.py` | **Seluruh pengolahan citra** (tanpa library): tahap A + tahap B, aturan, penjelasan setiap langkah. |
| `dip/dasar.py` | Materi 01/02: matematika dasar tanpa `math`, sampling, cropping. |
| `dip/m02_enhancement.py` | Materi 02: histogram, statistik, LUT stretching/gamma/equalization, threshold, slicing. |
| `dip/m03_spasial.py` | Materi 03: konvolusi 3x3, lowpass rata-rata NxN, std lokal, median 3x3, filter min/maks, Sobel. |
| `dip/m05_warna.py` | Materi 05: grayscale (Y dari YIQ), RGB→HSI, jarak warna Euclid. |
| `dip/m06_restorasi.py` | Materi 06: deteksi salt & pepper, estimasi noise di patch datar. |
| `dip/m08_segmentasi.py` | Materi 08 (+02): Otsu, pelabelan region N4/N8, region growing. |
| `dip/m09_representasi.py` | Materi 09: erosi/dilasi/opening/closing, hole filling, top-hat, chain code, keliling, momen. |
| `tampilan.py` | GUI (hanya tampilan). |
| `uji.py` | Alat uji otomatis terhadap `gambar/jawaban.txt`. |

Satu kanal citra disimpan sebagai list Python 1 dimensi baris demi baris: piksel (x, y) ada di
indeks `y * lebar + x`, tetangga kanan `+1`, tetangga bawah `+lebar` (konsep N4/N8, materi 02).

## Alur pengolahan (16 langkah di GUI)

### Tahap A — mencari letak blister (foto utuh, diperkecil)

1. **Foto asli & sampling** (01). Foto HP 12 MP terlalu besar untuk Python murni. Setiap blok
   f×f piksel diwakili rata-rata 4 titik sampel `g = [f(a,a)+f(a,b)+f(b,a)+f(b,b)]/4` — lowpass
   kecil sebelum sampling agar tidak aliasing.
2. **YIQ & kroma ternormalisasi** (05). `Y = 0.299R + 0.587G + 0.114B` (terang-gelap),
   I dan Q (warna). Kroma `c1 = 100·I/(Y+20)`, `c2 = 100·Q/(Y+20)` sehingga warna yang sama tetap
   mirip walau terkena bayangan. Latar (kayu, kain, kulit) berwarna **hangat**, foil & obat
   **abu-abu/kebiruan**. Warna latar = median pita tepi foto.
3. **Threshold Otsu** (02 histogram + 08 thresholding). T dipilih otomatis dari histogram jarak
   kroma ke latar (dan jarak RGB Euclid).
4. **ROI blister** (09 + 02). `bukan latar = (jarak kroma > T1) OR (jarak RGB > T2)` → opening 5×5 →
   closing 9×9 → hole filling → pelabelan N4, buang region kecil / yang menempel tepi foto.
   Tangan/jari ikut terbuang karena warnanya hangat seperti latar.
5. **Potong blister dari foto asli** (01). Kotak ROI dikalikan faktor sampling → koordinat foto
   asli → hanya bagian itu yang di-sampling (lebih detail dan lebih cepat).

### Tahap B — menghitung obat (potongan blister)

6. **Cek noise → restorasi** (06 + 03). Persentase piksel *impulse* (ekstrem & beda > 60 dari
   median 3×3) dan noise di 25% blok 8×8 terdatar. Median 3×3 / lowpass 1/32 hanya dijalankan
   **bila perlu** (di semua foto uji: tidak perlu → langkah ditandai *dilewati*).
7. **Grayscale & histogram** (05 + 02). ROI dihitung ulang di potongan; statistik histogram.
8. **Tekstur: std lokal** (03 + 08). `std = sqrt(mean5x5(f²) − mean5x5(f)²)`. Foil bertekstur,
   obat halus. Otsu membagi halus/kasar; piksel kasar = **contoh foil** → kecerahan foil `Yf` dan
   kroma foil dipakai sebagai pembanding di langkah berikutnya.
9. **Top-hat** (09 morfologi graylevel + 03 filter min/maks). `top-hat = f − max(min(f))` dengan SE
   lebih besar dari obat → seberapa terang piksel **di atas foil di sekitarnya** (tahan lampu tidak
   rata). Piksel gelap dinaikkan ke 0.85·Yf agar tulisan/kantong kosong tidak merusak perkiraan foil.
10. **Beda warna dari foil** (05). Jarak Euclid kroma ke kroma foil → obat kuning/krem/merah.
11. **Kandidat** (02 + 09). `ROI AND halus AND (top-hat > T OR (beda kroma > T AND cukup terang))`,
    opening 3×3, hole filling (huruf timbul di obat).
12. **Pelabelan, penilaian & region splitting** (02 + 08 + 09). Label N8. Setiap region diberi
    deskriptor (luas, keterisian & kelonjongan dari momen) dan dibandingkan dengan **cincin** di
    sekelilingnya (`dilasi besar − dilasi kecil` = dinding kantong/foil). Aturan:
    - luas wajar, bentuk rapi (keterisian ≥ 0.82, kelonjongan ≤ 5),
    - cincin berada di dalam blister dan berwarna foil (menolak tutup botol kuning, jari),
    - **obat putih**: lebih terang ≥ 15 dari cincin **dan** ≥ 1.14 × kecerahan foil,
    - **obat berwarna**: beda kroma ≥ 10 dari cincin,
    - bukan berwarna latar (kantong kosong yang menampakkan meja).

    Region yang gagal **dibelah** (region splitting, slide 08) dengan Otsu di dalam region pada
    fitur yang paling memisahkan (kecerahan, kroma, atau tekstur — dipilih dari η Otsu terbesar),
    lalu tiap bagian dinilai lagi (maks 3 tingkat).
13. **Gabung belahan kapsul** (09 closing + 05). Kapsul merah|putih terdeteksi sebagai dua region.
    Dua region berdekatan yang salah satunya berwarna disatukan dengan closing; dipakai hanya bila
    hasilnya berbentuk kapsul. Belahan berwarna yang sendirian dilengkapi belahan putihnya.
14. **Konsistensi ukuran** (09). Luas acuan = median kelompok luas yang paling dominan
    (0.6–1.67×). Serpihan < 0.4× dan benda > 2.2× acuan ditolak.
15. **Chain code & deskripsi bentuk** (09). Batas ditelusuri chain code 8 arah; keliling
    `P = #genap + √2·#ganjil`; kebundaran `4πA/P²`; kelonjongan → bulat / kaplet-oval / kapsul.
16. **Hasil**. Elips setara dari momen orde 2 (pusat, sumbu a & b, arah θ = ½·atan2(2μ11, μ20−μ02)).

## Teori di luar slide (perluasan yang diizinkan)

| Teknik | Sumber | Alasan | Kenapa bukan ML |
|---|---|---|---|
| Otsu | Gonzalez (Optimum Global Thresholding) | Memilih T otomatis untuk foto terang/gelap yang berbeda | Rumus tertutup dari histogram, tanpa data latih |
| Hole filling | Gonzalez (morfologi) | Huruf timbul di obat & obat di dalam ROI | Flood fill dari tepi |
| Top-hat graylevel | Gonzalez (morfologi graylevel) | "Lebih terang dari foil di sekitarnya" | Erosi/dilasi graylevel = filter min/maks (slide 03) |
| Momen & elips setara | Gonzalez (deskriptor region) | Keterisian, kelonjongan, arah | Rumus statistik koordinat piksel |
| Kroma ternormalisasi I/(Y+20) | Perluasan YIQ (slide 05) | Warna tidak terpengaruh bayangan | Pembagian sederhana |
| Region splitting dengan Otsu | Slide 08 (split & merge) + Otsu | Memisahkan obat yang menempel ke foil terang | Threshold di dalam region |
| Deret Taylor arctan | Matematika | Tidak boleh memakai `math` | Rumus deret |
| DFT / filter frekuensi (materi 04) | — | **Tidak dipakai**: foto uji tidak punya noise periodik, dan DFT Python murni untuk ~300 ribu piksel terlalu lambat | — |

## Hasil uji (18 foto di folder `gambar/`)

Kunci jawaban (`gambar/jawaban.txt`) = jumlah obat yang **terlihat** (obat yang tertutup jari /
benda lain tidak dihitung). Jalankan `python uji.py`.

| Foto | Kunci | Hasil | Catatan |
|---|---|---|---|
| IMG_…080312 | 6 | 6 | |
| IMG_…080320 | 5 | 5 | |
| IMG_…080332 | 0 | 0 | semua kantong kosong |
| IMG_…080346 | 4 | 4 | termasuk kaplet yang tertutup sobekan foil |
| IMG_…080352 | 4 | **3** | kaplet ke-4 hanya ~25% terlihat (sobekan foil) → ditolak sebagai serpihan |
| WA 11.18.23 | 10 | 10 | |
| WA 2.34.27 | 6 | 6 | |
| WA 3.36.48 | 3 | 3 | foto sangat gelap + jari |
| WA 3.47.07 (1) | 10 | **9** | satu obat di area bayangan tangan, foil memantulkan warna kain → keluar ROI |
| WA 3.47.07 | 11 | 11 | dua blister |
| WA 3.47.20 | 5 | 5 | |
| WA 3.49.44 | 12 | 12 | dua blister menumpuk, obat putih & krem |
| WA 3.51.08 | 4 | 4 | tutup botol kuning & benda hitam |
| WA 3.58.32 | 1 | 1 | kaplet kuning |
| WA 3.58.32 (2) | 2 | 2 | kapsul merah-putih |
| WA 3.58.32 (3) | 5 | 5 | kapsul merah-putih |
| WA 5.32.38 | 5 | 5 | kaplet kuning + kapsul, dua blister |
| WA 5.32.39 | 2 | 2* | *jumlah benar tetapi **kebetulan**: 2 pantulan plastik terdeteksi, kaplet & kapsul asli terlewat |

**16 dari 18 foto tepat** (total selisih 2 butir), rata-rata ±2 detik per foto. Per butir obat:
91 dari 95 obat terdeteksi, 2 deteksi palsu (keduanya di foto 5.32.39).

## Keterbatasan & saran pengambilan foto

- Foto **5.32.39** paling sulit: cahaya sangat terang, banyak kantong sobek dengan plastik
  mengkilap — pantulannya terang, halus, dan kompak seperti obat putih.
- Obat yang hanya sedikit terlihat (tertutup sobekan foil > ±60%) atau berada di **bayangan kuat**
  bisa terlewat.
- Ukuran minimal obat 100 piksel pada skala kerja (sisi terpanjang foto ≈ 640 px): blister
  jangan difoto terlalu jauh.
- Saran: foto dari atas, cahaya merata, latar polos yang warnanya beda dari foil (kayu/kain),
  tangan tidak menutupi kantong.

## Parameter penting (`deteksi.py`, bagian PARAMETER)

| Parameter | Nilai | Arti |
|---|---|---|
| `UKURAN_A` / `UKURAN_B` | 400 / 640 | skala tahap A (cari blister) / tahap B (hitung obat) |
| `JENDELA_STD` | 5 | jendela simpangan baku lokal |
| `BEDA_TERANG`, `TERANG_FOIL` | 15, 1.14 | syarat obat putih |
| `BEDA_KROMA` | 10 | syarat obat berwarna |
| `MIN_ISI`, `MAKS_LONJONG` | 0.82, 5.0 | syarat bentuk |
| `KECIL_ACUAN`, `BESAR_ACUAN` | 0.4, 2.2 | konsistensi ukuran |
| `MIN_LUAS` | 100 | luas minimal satu obat (piksel skala kerja) |
