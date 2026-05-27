import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import os


# KONFIGURASI HALAMAN & CSS 

st.set_page_config(page_title="CampusCALM", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS dark mode
st.markdown("""
    <style>
    /* Background utama aplikasi pekat */
    .stApp {
        background-color: #000000; /* True Black iOS */
        color: #FFFFFF;
    }
    
    /* Mengubah jenis font global */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Judul & Teks Utama */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #FFFFFF !important;
    }
    
    /* Info/Alert Box Glassmorphism style */
    .stAlert {
        background-color: #1C1C1E !important; /* Dark gray cell */
        border: 1px solid #2C2C2E !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* Styling Navigasi Tabs Mode Gelap */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1C1C1E;
        padding: 6px;
        border-radius: 14px;
        border: 1px solid #2C2C2E;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        border: none !important;
        color: #8E8E93 !important; /* Gray text untuk unselected */
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2C2C2E !important; /* Highlight tab aktif */
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    
    /* Tombol Utama iOS Blue */
    .stButton>button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        background-color: #0A84FF !important; /* iOS Dark Blue */
        color: white !important;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(10, 132, 255, 0.3);
    }
    .stButton>button:hover {
        background-color: #0066CC !important;
        transform: scale(0.99);
    }
    
    /* Dropdown & Input Form styling */
    div[data-baseweb="select"] > div {
        background-color: #1C1C1E !important;
        border: 1px solid #2C2C2E !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #E5E5EA !important;
    }
    
    /* Expander Container (Simulasi Section) */
    [data-testid="stExpander"] {
        background-color: #1C1C1E !important;
        border-radius: 16px !important;
        border: 1px solid #2C2C2E !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* Slider styling customization */
    .stSlider [data-testid="stTickBar"] {
        color: #8E8E93;
    }

    /* Hilangkan footer bawaan */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# INISIALISASI SESSION STATE

if 'is_analyzed' not in st.session_state:
    st.session_state.is_analyzed = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}


# INPUT MODEL & SCALER

@st.cache_resource
def load_ml_components():
    model_path = 'model/voting_model.pkl' if os.path.exists('model/voting_model.pkl') else 'voting_model.pkl'
    scaler_path = 'model/scaler.pkl' if os.path.exists('model/scaler.pkl') else 'scaler.pkl'
    return joblib.load(model_path), joblib.load(scaler_path)

try:
    model, scaler = load_ml_components()
except Exception as e:
    st.error(f"Gagal memuat komponen ML. Error: {e}")
    st.stop()

FEATURES = [
    "gender", "age", "stress_experience", "heartbeat_palpitations", "anxiety_tension",
    "sleep_problems", "restlessness", "headaches", "irritability", "concentration_problems",
    "sadness_low_mood", "health_issues", "loneliness_isolation", "academic_overload", 
    "peer_competition", "relationship_stress", "professor_difficulties", "work_environment", 
    "lack_relaxation_time", "home_environment", "low_academic_confidence", "subject_confidence", 
    "academic_conflicts", "class_attendance", "weight_changes"
]


# LAYOUT UTAMA & TABS INPUT

# Title
st.markdown("<h1 style='text-align: center; font-weight: 700; letter-spacing: -0.5px; margin-bottom: 5px;'>🧠 CampusCALM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8E8E93; margin-bottom: 2rem;'>Smart Predictions, Better Mental Health.</p>", unsafe_allow_html=True)

st.info("**💡 Panduan Pengisian Skala (1-5):** Nilai 1 mendeskripsikan kondisi paling minim / jarang terjadi, sedangkan nilai 5 mendeskripsikan kondisi paling maksimal / konstan.")

tabs = st.tabs(["👥 Demografi", "🧘‍♂️ Psikologis", "📚 Akademik", "🌍 Lingkungan & Sosial"])
user_input = {}
options = [1, 2, 3, 4, 5]

# TAB 1: DEMOGRAFI
with tabs[0]:
    user_input["gender"] = st.selectbox("Gender (0: Laki-laki, 1: Perempuan)", [0, 1])
    user_input["age"] = st.number_input("Umur saat ini (Tahun)", min_value=15, max_value=40, value=20)
    user_input["stress_experience"] = st.selectbox("Pernah Mengalami Stres Berat Sebelumnya?", options)

# TAB 2: PSIKOLOGIS
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        user_input["heartbeat_palpitations"] = st.selectbox("Jantung Sering Berdebar Kencang", options)
        user_input["anxiety_tension"] = st.selectbox("Tingkat Kecemasan / Ketegangan Pikiran", options)
        user_input["sleep_problems"] = st.selectbox("Masalah Pola Tidur / Insomnia", options)
        user_input["restlessness"] = st.selectbox("Merasa Gelisah Berlebihan", options)
    with col2:
        user_input["headaches"] = st.selectbox("Frekuensi Mengalami Sakit Kepala", options)
        user_input["irritability"] = st.selectbox("Tingkat Emosional / Gampang Marah", options)
        user_input["concentration_problems"] = st.selectbox("Kesulitan Fokus & Konsentrasi Belajar", options)
        user_input["sadness_low_mood"] = st.selectbox("Sering Merasa Sedih / Bad Mood", options)
        user_input["health_issues"] = st.selectbox("Keluhan Sakit Fisik Lainnya", options)
        user_input["weight_changes"] = st.selectbox("Perubahan Berat Badan Secara Drastis", options)

# TAB 3: AKADEMIK
with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        user_input["academic_overload"] = st.selectbox("Tekanan & Volume Tugas Kuliah", options)
        user_input["low_academic_confidence"] = st.selectbox("Rasa Kurang Percaya Diri terhadap Nilai", options)
        user_input["subject_confidence"] = st.selectbox("Tingkat Penguasaan Materi Kuliah", options)
    with col2:
        user_input["academic_conflicts"] = st.selectbox("Konflik Akademik (Dosen / Universitas)", options)
        user_input["class_attendance"] = st.selectbox("Tingkat Kehadiran Presensi Kelas", options)
        user_input["professor_difficulties"] = st.selectbox("Hambatan Komunikasi dengan Dosen", options)

# TAB 4: LINGKUNGAN & SOSIAL 
with tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        user_input["loneliness_isolation"] = st.selectbox("Merasa Kesepian atau Terisolasi", options)
        user_input["peer_competition"] = st.selectbox("Tekanan Kompetisi Antar Teman Sekelas", options)
        user_input["relationship_stress"] = st.selectbox("Stres Hubungan (Asmara / Keluarga)", options)
    with col2:
        user_input["work_environment"] = st.selectbox("Kondisi Lingkungan Kerja / Tugas Kelompok", options)
        user_input["home_environment"] = st.selectbox("Kenyamanan Lingkungan Tempat Tinggal/Kos", options)
        user_input["lack_relaxation_time"] = st.selectbox("Kurangnya Waktu untuk Liburan/Relaksasi", options)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Menyimpan input ke session state 
    st.session_state.user_input = user_input.copy()
    
    # Tombol lihat analisis
    if st.button("Tampilkan Hasil Analisis", type="primary", use_container_width=True):
        st.session_state.is_analyzed = True

    # SELURUH BLOK HASIL DIKUNCI DI SINI 
    if st.session_state.is_analyzed:
        st.markdown("<br><hr style='border-color: #2C2C2E;'>", unsafe_allow_html=True)
        
        # Hitung Prediksi
        input_df = pd.DataFrame([st.session_state.user_input])[FEATURES]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        
        status = "Distress (Stres Tinggi)" if prediction == 0 else "Normal (Terkendali)"
        color = "#FF453A" if prediction == 0 else "#30D158" # iOS Neon Red / Neon Green
        
        st.markdown(f"<h2 style='text-align: center; color: {color} !important;'>Status: {status}</h2>", unsafe_allow_html=True)
        
        # Rekomendasi solusi
        recommendations = []
        if st.session_state.user_input["academic_overload"] >= 4 and st.session_state.user_input["sleep_problems"] >= 4:
            recommendations.append("Beban belajar dan gangguan tidurmu berada di zona merah. Terapkan batasan tegas kapan harus berhenti belajar demi menjaga tubuh.")
        if st.session_state.user_input["loneliness_isolation"] >= 4:
            recommendations.append("Interaksi sosialmu minim. Sempatkan menyapa teman lama atau ikut berdiskusi langsung di sekretariat/lingkungan kampus.")
        if st.session_state.user_input["lack_relaxation_time"] >= 4:
            recommendations.append("Tubuhmu butuh rehat. Jadwalkan waktu kosong minimal 45 menit tanpa melihat notifikasi tugas sama sekali.")
        if not recommendations:
            recommendations.append("Kombinasi pola hidup dan aktivitas akademikmu sudah berada pada jalur yang sehat. Pertahankan keseimbangan ini!")

        st.markdown("#### 💡 Actionable Insights")
        for rec in recommendations:
            st.markdown(f"- {rec}")
            
        st.markdown("<br>", unsafe_allow_html=True)

        # Visualisasi Diagram  
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**📊 Faktor Pemicu Tertinggi**")
            filtered_metrics = {k: v for k, v in st.session_state.user_input.items() if k not in ["gender", "age"]}
            top_factors = sorted(filtered_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Setup Bar Chart  
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor('#1C1C1E')
            ax.set_facecolor('#1C1C1E')
            
            bars = ax.barh([x[0].replace("_", " ").title() for x in top_factors], 
                    [x[1] for x in top_factors], color='#0A84FF', height=0.55)
            
            ax.set_xlim(0, 5)
            ax.invert_yaxis()
            ax.tick_params(colors='#FFFFFF', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#2C2C2E')
            ax.grid(axis='x', color='#2C2C2E', linestyle='--', alpha=0.7)
            st.pyplot(fig)

        with col_chart2:
            st.markdown("**🕸️ Radar Keseimbangan Mental**")
            radar_features = ["sleep_problems", "academic_overload", "anxiety_tension", "peer_competition", "relationship_stress"]
            user_vals = [st.session_state.user_input[f] for f in radar_features]
            avg_vals = [3.1, 3.4, 2.9, 3.0, 2.6] # Representasi baseline rata-rata
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=user_vals, theta=[f.replace("_", " ").title() for f in radar_features], fill='toself', name='Skormu', line_color='#0A84FF'))
            fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=[f.replace("_", " ").title() for f in radar_features], fill='toself', name='Rata-rata', line_color='rgba(142, 142, 147, 0.4)'))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5], gridcolor='#2C2C2E', tickfont=dict(color='#8E8E93')),
                    angularaxis=dict(gridcolor='#2C2C2E', tickfont=dict(color='#FFFFFF', size=10)),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                legend=dict(font=dict(color='#FFFFFF')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=25, b=25)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # SIMULASI WHAT-IF
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔄 Simulasi Modifikasi Kebiasaan (Live Preview)", expanded=True):
            st.markdown("<p style='color: #8E8E93; font-size: 0.9em; margin-bottom: 1.5rem;'>Geser slider di bawah ini untuk melihat simulasi perbaikan gaya hidupmu secara instan.</p>", unsafe_allow_html=True)
            
            sim_sleep_quality = st.slider("Kualitas Tidur (1 = Insomnia Parah, 5 = Sangat Nyenyak)", 1, 5, 3)
            sim_load_management = st.slider("Manajemen Beban Tugas (1 = Keteteran, 5 = Terkendali Baik)", 1, 5, 3)
            
            simulated_data = st.session_state.user_input.copy()
            
            # Semakin besar slider (makin bagus), angka masalah di dataset asli semakin kecil (6 - input)
            simulated_data["sleep_problems"] = 6 - sim_sleep_quality
            simulated_data["academic_overload"] = 6 - sim_load_management
            
            sim_df = pd.DataFrame([simulated_data])[FEATURES]
            sim_scaled = scaler.transform(sim_df)
            sim_pred = model.predict(sim_scaled)[0]
            
            sim_status = "Distress (Stres Tinggi)" if sim_pred == 0 else "Normal (Aman & Terkendali)"
            sim_color = "#FF453A" if sim_pred == 0 else "#30D158"
            
            st.markdown(f"<h4 style='text-align: center; margin-top: 1rem;'>Prediksi Simulasi: <span style='color:{sim_color}; font-weight:700;'>{sim_status}</span></h4>", unsafe_allow_html=True)
            
            if sim_pred == 1:
                st.success("✨ Hasil Bagus! Skenario perbaikan gaya hidup ini terbukti efektif menurunkan tingkat risiko stres menurut kalkulasi algoritma.")
            else:
                st.warning("⚠️ Perubahan opsi ini dirasa belum cukup dominan. Cobalah naikkan kualitas tidur atau kombinasikan dengan penyeimbang manajemen waktu lainnya.")