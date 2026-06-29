# **CampusCALM**

> **Final Project Machine Learning**

Mahasiswa sering mengalami tekanan akademik, beban tugas, kurang tidur, kecemasan, serta tekanan sosial yang dapat meningkatkan tingkat stres. Tingkat stres yang tinggi dapat berdampak negatif terhadap kesehatan mental, konsentrasi belajar, produktivitas akademik dan performa mahasiswa. Maka dari itu kami membuat aplikasi berbasis AI dan Machine Learning CampusCALM yang digunakan untuk mengidentifikasi pola stress, Memprediksi tingkat stress mahasiswa.

---

> ## **Dataset**
Sesuai namanya berada di dalam folder data, ini adalah folder untuk menyimpan semua dataset yang digunakan dalam proyek berupa .csv.
* **Title**: Stress Indicator Dataset for Mental Health Classification
* **Dataset Source**: https://doi.org/10.17632/2gsjv8m7ch.1
* **Dataset Description**: kumpulan data yang berisi 2000 responden mahasiswa dengan 25 fitur yang merepresentasikan berbagai faktor penyebab stres. Dataset ini mencakup lima kategori utama, yaitu faktor psikologis, fisiologis, sosial, lingkungan, dan akademik, seperti tingkat kecemasan, kualitas tidur, beban belajar, hingga dukungan sosial.

---

> ## **Model**
Folder ini adalah tempat untuk menyimpan model machine learning yang sudah selesai dilatih yang berupa notebook .ipynb.
| Model | Keterangan |
|---|---|
| Logistic Regression | Model Baseline |
| Random Forest | Estimators & Feature Importance |
| Support Vector Machine | Kernel Linear |
| K-Nearest Neighbors | Default Hyperparameter |
| **Ensemble (Majority Voting)** | **Gabungan Random Forest + Support Vector Machine + K-Nearest Neighbors (Model Final)** |

---

> ## **Analisis**

### Kondisi Data
Dataset awal memiliki **class imbalance** yang signifikan:

| Kelas | Jumlah |
|---|---|
| Eustress (1) | 1828 |
| Other/Mixed (2) | 102 |
| Distress (0) | 70 |

Mayoritas data termasuk Eustress, sehingga model bisa bias jika tidak ditangani.
Selain itu, fitur `age` memiliki outlier ekstrem di luar rentang 18–22 tahun yang perlu dibersihkan.

**Penanganan:**
- Filtering usia: hanya data dengan `age >= 18` dan `age <= 22` yang dipertahankan
- SMOTE digunakan untuk menyeimbangkan kelas menjadi **1.357 data per kelas**
- StandardScaler untuk normalisasi seluruh fitur

---

### Hasil Evaluasi Model

| Model | Accuracy | Precision | Recall | F1 (Default) | F1 (Tuned) |
|---|---|---|---|---|---|
| Logistic Regression | 98,9% | 99,1% | 98,9% | 98,97% | 98,97% |
| SVM | 98,9% | 98,9% | 98,9% | 98,92% | 98,89% |
| Random Forest | 98,6% | 98,7% | 98,6% | 98,54% | 97,99% |
| KNN | 91,6% | 95,9% | 91,6% | 92,89% | 98,95% |
| **Ensemble (Voting)** | **98,9%** | **98,9%** | **98,9%** | **98,92%** | **99,15%** |

Model **Ensemble Voting Classifier** dipilih sebagai model final karena performanya paling stabil dan konsisten.
Pendekatan ini menggabungkan kekuatan LR + RF + SVM + KNN sehingga kelemahan masing-masing model bisa saling ditutupi.
Setelah hyperparameter tuning dengan GridSearchCV, F1-Score Ensemble meningkat menjadi **99,15%**.

---

> ## **Cara Menjalankan**
Berikut merupakan link demo deplyoment aplikasi "CampusCalm" kami: https://campuscalm.streamlit.app/
atau kamu juga bisa menjalankannya secara local:
### 1. Install Dependencies
Pastikan Python 3.8+ terinstal, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 2. Training Model
Melatih ulang seluruh model klasifikasi menggunakan Ensemble (Majority Voting):
```bash
# Jalankan di dalam Visual Studio Code atau Google Colab untuk mendapatkan .pkl.
```

### 3. Deployment (Streamlit)
Jalankan dashboard aplikasi menggunakan Streamlit:
```bash
python -m streamlit run app.py
```
Aplikasi akan berjalan pada alamat local: [http://localhost:8501](http://localhost:8501)

---
