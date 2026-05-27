# Menggunakan base image Python yang ringan
FROM python:3.9-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
# Pastikan ada file Requirements.txt di direktori yang sama dengan Dockerfile
COPY requirements.txt .

# Menginstal dependensi Python
# --no-cache-dir digunakan agar image Docker tidak membengkak
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh source code dan model (termasuk folder data/ dan model/) ke dalam container
COPY . .

# Membuka port default yang digunakan oleh Streamlit
EXPOSE 8501

# Command untuk menjalankan aplikasi Streamlit saat container di-start
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]