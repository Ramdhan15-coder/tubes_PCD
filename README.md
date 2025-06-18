# Aplikasi Pengolahan Citra Digital dengan Deteksi Objek Pada Video Game

Aplikasi desktop ini dikembangkan sebagai proyek akhir mata kuliah Pengolahan Citra Digital. Aplikasi ini dibangun menggunakan Python dengan library utama PyQt5 untuk antarmuka dan OpenCV untuk pemrosesan citra.

Tujuan utama dari aplikasi ini adalah untuk menyediakan platform interaktif guna mengeksplorasi berbagai teknik PCD, mulai dari perbaikan kualitas citra dasar hingga implementasi metode deteksi objek modern.

## Fitur Utama

- **Perbaikan Kualitas Citra:**
  - Berbagai filter spasial (Penajaman, Penghalusan, Deteksi Tepi).
  - Penyesuaian kontras (Histogram Equalization, CLAHE).
  - Transformasi warna dan filter frekuensi (FFT).

- **Ekstraksi Ciri:**
  - Analisis statistik warna (Mean, Standar Deviasi).
  - Deteksi titik kunci menggunakan algoritma ORB.

- **Deteksi Objek:**
  - **Deteksi Wajah:** Menggunakan metode klasik Haar Cascade Classifier pada gambar statis.
  - **Deteksi Multi-Objek:** Menggunakan model Deep Learning YOLOv4-tiny (pre-trained on COCO dataset) untuk mendeteksi 80 kelas objek secara real-time pada gambar statis maupun video.
