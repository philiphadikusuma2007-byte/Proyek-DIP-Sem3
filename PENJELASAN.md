# Deteksi Sisa Obat dalam Blister — Pengolahan Citra Digital

Program menghitung **jumlah obat yang masih tersisa** di dalam kemasan blister dari
sebuah foto. Seluruh pengolahan memakai konsep **Pengolahan Citra Digital** dari
materi kuliah. Program ini **tidak memakai Machine Learning sama sekali**: tidak ada
pelatihan, model, atau dataset latih. Setiap keputusan berupa rumus dan aturan
(threshold) yang bisa dijelaskan.

## 1. Aturan library

| File | Isi | Library |
|---|---|---|
| `citra.py` | semua rumus pengolahan citra | **tidak ada** (bahkan `math` tidak dipakai) |
| `deteksi.py` | alur deteksi A–F | `time` (mengukur durasi), `PIL` **hanya** di `baca_foto()` untuk membuka file JPEG menjadi byte R,G,B |
| `main.py` | tampilan (UI) | `tkinter`, `threading`, `ctypes` (DPI layar). Gambar ditampilkan lewat format PPM yang disusun manual, jadi PIL tidak dipakai untuk tampilan |
| `uji_dataset.py` | uji otomatis ke semua foto | `os`, `sys` |

Citra disimpan sebagai **list datar** Python (indeks `i = y*lebar + x`). Semua operasi
ditulis manual: konversi warna, histogram, Otsu, filter, morfologi, pelabelan, momen,
dan kode rantai. Akar kuadrat ditulis `x ** 0.5`, dan π ditulis sebagai angka.

## 2. Cara menjalankan

```
cd D:\DIP\projectsAkhir\Proyek-DIP-Sem3
python main.py              # aplikasi berjendela
python uji_dataset.py       # uji semua foto di folder gambar/ (bandingkan dgn jawaban.txt)
python uji_dataset.py -v    # + rincian deskriptor setiap region
```

Kebutuhan: Python 3 dan Pillow (`pip install pillow`, hanya untuk membuka file JPEG).

Di aplikasi:
1. Klik **Pilih Foto Blister...**, lalu klik **DETEKSI**.
2. Hasil yang ditampilkan:
   - **Tab Hasil:** obat diberi kontur hijau dan nomor (`*` menandai potongan obat). Region yang ditolak diberi kontur merah beserta alasannya.
   - **Tab Tahapan Pengolahan:** 12 citra antara dari setiap tahap.
   - **Tab Keputusan per Region:** tabel deskriptor setiap region (luas, keliling, kekompakan, isi elips, jarak warna, dll.) beserta keputusannya.
   - **Panel kiri:** jumlah sisa obat dan daftar perbaikan citra yang dipilih otomatis.

## 3. Alur program

```
Foto ─► A. cari blister (kasar) ─► B. potong dari foto asli ─► C. perbaikan citra
      ─► D. mask blister + warna pembanding ─► E. calon obat + region growing
      ─► F. penilaian region (deskriptor) ─► JUMLAH SISA OBAT
```

### A. Mencari blister secara kasar (`cari_blister_kasar`, `segmentasi_blister`)
Foto diperkecil manual (sampling) sampai sisi terpanjangnya 400 px. Ciri foil:
- **Warna kebiruan.** Komponen **Cb** dari model YCbCr: `Cb = 128 − 0.169R − 0.331G + 0.5B`.
  Foil perak memantulkan cahaya kebiruan (Cb tinggi), sedangkan meja kayu kekuningan (Cb rendah).
- **Tekstur embos.** Energi **highpass** `|f − lowpass(f)|` dirata-rata di jendela 9×9 lalu
  dibagi kecerahan lokal.

Kedua peta di-threshold dengan **Otsu**. Program mengukur **keterpisahan Otsu**
η = σ²antar-kelas / σ²total pada Cb:
- **η ≥ 0,85** (meja polos, cahaya baik): cukup ciri warna.
- **η < 0,85** (foto gelap / latar ramai): foil = Cb tinggi **dan** bertekstur.

Setelah itu: **closing → isi lubang → opening → komponen terhubung terbesar**. Bounding box-nya diperlebar 20%.

### B. Memotong blister dari foto asli (`sampling_ulang`)
Daerah blister diambil dari foto **resolusi penuh** lalu diperkecil ke 480 px. Obat jadi
cukup besar, terutama untuk foto 2600×4624 yang blisternya kecil.

### C. Perbaikan citra otomatis (`perbaiki_citra`)
Program memilih sendiri materi yang dipakai berdasarkan statistik citra:

| Kondisi yang diukur | Materi yang dipakai |
|---|---|
| > 0,4% piksel ekstrem terisolasi (salt & pepper) | **Median filter 3×3** |
| residu noise di blok paling datar > 4 | **Lowpass filter 3×3** (bobot 1-2-1) |
| rata-rata keabuan < 90 (gelap) | **Gamma correction** `s = 255·(r/255)^γ`, γ = 0,5 / 0,7 |
| rentang persentil 1–99% < 60 | **Histogram equalization** `s = 255·CDF(r)` |
| simpangan baku < 45 (kontras rendah) | **Contrast stretching** `[p1..p99] → [0..255]` |

### D. Mask blister dan warna pembanding
- **Mask blister di potongan:** langkah A diulang di potongan, lalu digabung (union) dengan ROI tahap A dan lubangnya diisi. Pembungkus **oktagon** (min/maks x, y, x+y, x−y) menutup coakan di tepi.
- **Warna meja:** median warna di luar blister.
- **Warna foil lokal:** **median per blok 24×24** dari piksel foil (order statistic). Median dipakai karena tahan terhadap kilau tepi kantong. Bayangan di atas foil juga ikut terkompensasi karena dihitung per daerah.

### E. Calon obat (`mirip_obat`, region growing)
Sebuah piksel disebut **mirip obat** jika:
- jarak warna RGB (Euclid) ke foil lokal > 40, **dan**
- bukan warna meja (jarak > 45), **dan** salah satu dari:
  - tidak lebih gelap dari foil **dan** kurang biru dari foil. Kromatisitas
    `b = B/(R+G+B)`; foil dan plastik kantong sama-sama kebiruan, permukaan obat tidak.
  - ≥ 1,25× lebih terang dari foil (obat putih di bawah cahaya kebiruan).
  - **warna mencolok** (saturasi HSI > 0,45 dan V = maks(R,G,B) cukup tinggi), misalnya kapsul merah.

Langkah berikutnya:
1. **Biji (seed)** = mirip obat **dan halus** (std lokal 7×7 < 12).
2. **Region growing:** biji di-dilasi 3×3 sebanyak 4 kali, tetapi hanya ke piksel yang
   memenuhi syarat yang lebih longgar. Tepi obat dan garis sambungan kapsul dua warna
   ikut terisi, sehingga 1 kapsul = 1 region.
3. **Closing, isi lubang, opening**, lalu **pelabelan komponen terhubung** (8-tetangga).

### F. Penilaian setiap region (deskriptor)
Untuk setiap region dihitung:
- **Momen:** pusat massa, momen pusat μ20, μ02, μ11, lalu sumbu elips λ1, λ2.
  - *lonjong* = √(λ1/λ2)
  - *isi elips* = luas / (4π√(λ1λ2)) → ≈ 1 untuk bulat/kapsul utuh
- **Kode rantai** 8-arah → keliling → *kekompakan* = 4π·luas / keliling².
- Warna rata-rata, jarak ke meja, kromatisitas (r, g), jarak ke foil, selisih biru,
  kecerahan relatif, bagian putih jenuh, dan bagian yang halus.
- **Cincin** di sekeliling region (dilasi 11 − dilasi 3): berapa bagiannya yang berwarna meja.

Aturan penolakan:

| Alasan | Arti |
|---|---|
| `luar` | < 60% region berada di dalam blister |
| `kosong` | warna atau kromatisitas sama dengan meja → kantong kosong (plastik bening, meja terlihat) |
| `di meja` | > 35% cincinnya berwarna meja → pantulan di meja |
| `kasar` | < 35% bagiannya halus → plastik kusut |
| `foil` / `plastik` | warnanya sama dengan foil (sama-sama kebiruan) |
| `pantulan` | ≥ 85% putih jenuh (R, G, B ≥ 245) |
| `bentuk` | isi elips < 0,70 atau lonjong > 4,5 |

Region yang gagal karena `bentuk`/`luar` **dipecah dengan opening bertingkat** (9, 13, 17)
untuk memutus "jembatan" ke silau atau plastik, lalu setiap bagiannya dinilai ulang.

**Aturan ukuran:** obat dalam satu blister berukuran sama. Patokan 1 butir = median luas
region besar. Region dengan luas < 0,30× patokan dianggap **serpihan** (tidak dihitung).
Region dengan luas < 0,70× patokan dihitung sebagai **potongan obat**.

## 4. Pemetaan ke materi kuliah

| Materi (PPT) | Dipakai untuk | Fungsi |
|---|---|---|
| 01–02 Introduction | sampling / resize manual, ketetanggaan 4 & 8 | `sampling_ulang`, `label_komponen` |
| 02 Image Enhancement | histogram, statistik, Otsu, gamma, stretching, equalization | `histogram`, `statistik`, `otsu`, `lut_*` |
| 03 Spatial Filtering | lowpass/mean, median, highpass, sliding window, perlebar tepi | `filter_rata`, `konvolusi`, `filter_median`, `perlebar_tepi` |
| 05 Color Processing | grayscale, YCbCr (Cb), HSI (S), HSV (V), normalized RGB | `ke_abu`, `ke_cb`, `saturasi_hsi`, `jarak_kromatis` |
| 06 Restoration | model noise (salt & pepper, gaussian), order statistic | `ukur_salt_pepper`, `ukur_noise`, `median_per_blok` |
| 08 Segmentation | threshold Otsu, region growing, komponen terhubung | `segmentasi_blister`, tahap E |
| 09 Representation | dilasi, erosi, opening, closing, isi lubang, kode rantai, momen | `dilasi`, `erosi`, `kode_rantai`, `momen` |

Materi frekuensi (DFT) dan kompresi (DCT) tidak dipakai, karena foto-foto ini tidak
mengandung noise periodik dan kompresi tidak relevan untuk menghitung obat.

## 5. Hasil uji pada dataset (`python uji_dataset.py`)

Jawaban manual ada di `gambar/jawaban.txt`.

| Foto | Kondisi | Jawaban | Program | |
|---|---|---|---|---|
| IMG_20261109_080312 | 6 tablet putih, 4 kantong kosong | 6 | 6 | benar |
| IMG_20261109_080320 | 5 tablet, 1 kantong kosong | 5 | 5 | benar |
| IMG_20261109_080332 | 4 kantong kosong semua | 0 | 0 | benar |
| IMG_20261109_080346 | 3 tablet lonjong + ½ tablet | 4 | 2 | salah |
| IMG_20261109_080352 | sama, sudut berbeda | 4 | 3 | salah |
| WA 11.14.53 PM (1) | foto gelap, 1 kapsul krem, pantulan lampu di meja | 1 | 1 | benar |
| WA 11.14.53 PM | foto buram & miring, 2 tablet | 2 | 2 | benar |
| WA 11.14.54 PM | 10 kapsul merah-putih | 10 | 10 | benar |
| WA 11.14.55 PM (1) | 1 kapsul, 9 kantong kosong kusut | 1 | 2 | salah |
| WA 11.14.55 PM | sama, sudut berbeda | 1 | 1 | benar |
| WA 11.18.23 PM | 10 tablet krem, foto gelap | 10 | 10 | benar |

**Akurasi: 8 dari 11 foto.** Waktu proses sekitar 3–10 detik per foto (Python murni, tanpa numpy).

### Penyebab kesalahan (keterbatasan)
- **080346 / 080352:** cahaya ruangan kebiruan membuat tablet putih ikut kebiruan, sehingga
  warnanya hampir sama dengan foil perak di sekitarnya (selisih b ≈ 0,01). Setengah butir di
  kantong kanan-bawah juga tertutup plastik kusut. Program memilih **tidak menebak**. Di versi
  sebelumnya angka 4 sempat "benar", tetapi karena kebetulan: yang terhitung adalah kantong
  kosong, sementara tablet aslinya terlewat.
- **WA 11.14.55 PM (1):** bagian putih kapsul menyatu dengan silau flash di tepi blister,
  sehingga hanya bagian merahnya yang terdeteksi. Serpihan kulit kapsul merah di kantong
  kosong lain jadi berukuran mirip dan ikut terhitung.
- **Umum:** obat berwarna biru pucat sulit dibedakan dari foil. Kamera sebaiknya tegak lurus di
  atas blister, dengan cahaya merata dan tanpa flash langsung.

## 6. Struktur folder

```
Proyek-DIP-Sem3/
├── main.py           tampilan (tkinter)
├── deteksi.py        alur deteksi A–F + semua parameter threshold
├── citra.py          rumus pengolahan citra (tanpa library)
├── uji_dataset.py    uji otomatis ke dataset
├── PENJELASAN.md     dokumen ini
├── gambar/           foto dataset + jawaban.txt
└── MATERI DIP/       materi kuliah (PPT)
```
