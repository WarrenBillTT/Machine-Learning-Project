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
