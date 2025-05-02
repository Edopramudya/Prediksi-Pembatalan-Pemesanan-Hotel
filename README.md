# Prediksi-Pembatalan-Pemesanan-Hotel
Proyek ini merupakan implementasi model machine learning untuk memprediksi apakah suatu pemesanan hotel akan dibatalkan (cancel) atau tidak berdasarkan data historis pelanggan. Aplikasi dibangun menggunakan Streamlit dan dapat diakses secara publik melalui link berikut:

https://prediksi-pembatalan-pemesanan-hotel.streamlit.app/

🔍 Tujuan Proyek
Tujuan dari proyek ini adalah untuk membantu pihak hotel dalam mengantisipasi kemungkinan pembatalan pemesanan, sehingga mereka dapat melakukan tindakan proaktif seperti overbooking atau penawaran ulang kamar. Dengan model prediksi yang akurat, manajer hotel dapat mengambil keputusan strategis yang lebih baik untuk meminimalkan potensi kerugian.

🧠 Model Machine Learning
Model utama yang digunakan dalam proyek ini adalah Random Forest Classifier, karena:
- Memiliki performa yang baik untuk data tabular.
- Mampu menangani fitur yang bersifat numerik maupun kategorikal.
- Lebih stabil dan minim overfitting dibanding Decision Tree tunggal.
- Model dilatih menggunakan dataset yang berisi informasi seperti:
- Lama tinggal (weekday/weekend)
- Tipe pelanggan
- Jumlah orang dewasa dan anak
- Tipe kamar
- Dan fitur lainnya

Evaluasi Model:
Akurasi: 78%
Presisi: 83.33%
Recall: 72.12%
F1-Score: 77.32%

🛠️ Teknologi yang Digunakan
- Python
- Pandas & Numpy
- Scikit-learn (Random Forest)
- Streamlit (untuk antarmuka web)
- Matplotlib & Seaborn (visualisasi data)
