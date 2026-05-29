import os
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# === KONFIGURASI HALAMAN ===
st.set_page_config(page_title="CampusCALM", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #FFFFFF !important; }
    .stAlert {
        background-color: #1C1C1E !important;
        border: 1px solid #2C2C2E !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #1C1C1E;
        padding: 6px; border-radius: 14px; border: 1px solid #2C2C2E;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 20px;
        border: none !important; color: #8E8E93 !important; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2C2C2E !important;
        color: #FFFFFF !important; box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .stButton>button {
        border-radius: 14px !important; font-weight: 600 !important;
        padding: 0.6rem 2rem !important; border: none !important;
        background-color: #0A84FF !important; color: white !important;
        transition: all 0.2s ease; box-shadow: 0 4px 12px rgba(10,132,255,0.3);
    }
    .stButton>button:hover { background-color: #0066CC !important; transform: scale(0.99); }
    div[data-baseweb="select"] > div {
        background-color: #1C1C1E !important; border: 1px solid #2C2C2E !important;
        border-radius: 12px !important; color: #FFFFFF !important;
    }
    div[data-testid="stMarkdownContainer"] p { color: #E5E5EA !important; }
    [data-testid="stExpander"] {
        background-color: #1C1C1E !important; border-radius: 16px !important;
        border: 1px solid #2C2C2E !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stSlider [data-testid="stTickBar"] { color: #8E8E93; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# === KONSTANTA & DATA MODEL ===
CLASS_LABELS = {0: "Distress (Stres Negatif)", 1: "Eustress (Stres Positif)", 2: "Other / Mixed Stress"}
CLASS_COLORS = {0: "#FF453A", 1: "#30D158", 2: "#FFD60A"}

CLASS_DESC = {
    0: "Kondisimu menunjukkan tanda-tanda **distress** — stres yang berdampak negatif pada kesehatan dan performa. Perlu perhatian serius.",
    1: "Kondisimu tergolong **eustress** — tekanan yang masih produktif dan mendorong pertumbuhan. Pertahankan keseimbangan ini.",
    2: "Kondisimu berada di zona **campuran** — ada kombinasi tekanan positif dan negatif. Perhatikan faktor-faktor pemicu di bawah.",
}

# Label & arah fitur untuk slider simulasi (True = fitur positif, higher = better)
FEATURE_META = {
    "stress_experience": ("Pengalaman Stres Berat", False),
    "heartbeat_palpitations": ("Jantung Berdebar Kencang", False),
    "anxiety_tension": ("Kecemasan / Ketegangan", False),
    "sleep_problems": ("Kualitas Tidur", False),
    "restlessness": ("Kegelisahan", False),
    "irritability": ("Emosional / Mudah Marah", False),
    "sadness_low_mood": ("Perasaan Sedih / Bad Mood", False),
    "loneliness_isolation": ("Kesepian / Isolasi Sosial", False),
    "concentration_problems": ("Kesulitan Konsentrasi", False),
    "headaches": ("Frekuensi Sakit Kepala", False),
    "health_issues": ("Keluhan Fisik", False),
    "weight_changes": ("Perubahan Berat Badan", False),
    "academic_overload": ("Beban Tugas Kuliah", False),
    "peer_competition": ("Kompetisi Antar Teman", False),
    "low_academic_confidence": ("Kurang Percaya Diri Akademik", False),
    "subject_confidence": ("Penguasaan Materi Kuliah", True),
    "academic_conflicts": ("Konflik Akademik", False),
    "class_attendance": ("Kehadiran Kelas", True),
    "professor_difficulties": ("Hambatan dengan Dosen", False),
    "work_environment": ("Kondisi Lingkungan Kelompok", False),
    "home_environment": ("Kenyamanan Tempat Tinggal", True),
    "relationship_stress": ("Stres Hubungan", False),
    "lack_relaxation_time": ("Kurang Waktu Relaksasi", False),
}

KATEGORI_FEATURES = {
    "Demographics": ["gender", "age"],
    "Emotional & Stress Indicators": ["stress_experience", "heartbeat_palpitations", "anxiety_tension", "sleep_problems", "restlessness", "irritability", "sadness_low_mood", "loneliness_isolation", "concentration_problems"],
    "Physical & Health Indicators": ["headaches", "health_issues", "weight_changes"],
    "Academic & Environment Stressors": ["academic_overload", "peer_competition", "low_academic_confidence", "subject_confidence", "academic_conflicts", "class_attendance", "professor_difficulties", "work_environment", "home_environment"],
    "Social & Relationship Factors": ["relationship_stress", "lack_relaxation_time"],
}

# === LOAD MODEL ===
@st.cache_resource
def load_ml_components():
    def resolve(name):
        return f"model/{name}" if os.path.exists(f"model/{name}") else name
    return (
        joblib.load(resolve("voting_model.pkl")),
        joblib.load(resolve("scaler.pkl")),
        joblib.load(resolve("feature_names.pkl")),
    )

@st.cache_resource
def get_top_features(_model, _features, n=5, exclude=("gender", "age")):
    """Top-N fitur berdasarkan importance RF dalam VotingClassifier."""
    imp = pd.Series(_model.estimators_[0].feature_importances_, index=_features)
    return imp.drop(list(exclude), errors="ignore").nlargest(n).index.tolist()

try:
    model, scaler, FEATURES = load_ml_components()
except Exception as e:
    st.error(f"Gagal memuat komponen ML. Error: {e}")
    st.stop()

# === SESSION STATE ===
if "is_analyzed" not in st.session_state:
    st.session_state.is_analyzed = False
if "user_input" not in st.session_state:
    st.session_state.user_input = {}

# === HEADER ===
st.markdown("<h1 style='text-align:center; font-weight:700; letter-spacing:-0.5px; margin-bottom:5px;'>CampusCALM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#8E8E93; margin-bottom:2rem;'>Smart Predictions, Better Mental Health.</p>", unsafe_allow_html=True)
st.info("**Panduan Pengisian Skala (1-5):** Nilai 1 mendeskripsikan kondisi paling minim / jarang terjadi, sedangkan nilai 5 mendeskripsikan kondisi paling maksimal / konstan.")

# === FORM INPUT ===
tabs = st.tabs(["Demographics", "Emotional & Stress Indicators", "Physical & Health Indicators", "Academic & Environment Stressors", "Social & Relationship Factors"])
ui = {}

with tabs[0]:
    ui["gender"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan")
    ui["age"]    = st.number_input("Umur saat ini (Tahun)", min_value=18, max_value=22, value=20)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        ui["stress_experience"] = st.slider("Pernah Mengalami Stres Berat Sebelumnya?", 1, 5, 3)
        ui["heartbeat_palpitations"] = st.slider("Jantung Sering Berdebar Kencang", 1, 5, 3)
        ui["anxiety_tension"] = st.slider("Tingkat Kecemasan / Ketegangan Pikiran", 1, 5, 3)
        ui["sleep_problems"] = st.slider("Masalah Pola Tidur / Insomnia", 1, 5, 3)
        ui["restlessness"]           = st.slider("Merasa Gelisah Berlebihan", 1, 5, 3)
    with c2:
        ui["irritability"] = st.slider("Tingkat Emosional / Gampang Marah", 1, 5, 3)
        ui["sadness_low_mood"] = st.slider("Sering Merasa Sedih / Bad Mood", 1, 5, 3)
        ui["loneliness_isolation"] = st.slider("Merasa Kesepian atau Terisolasi", 1, 5, 3)
        ui["concentration_problems"] = st.slider("Kesulitan Fokus & Konsentrasi Belajar", 1, 5, 3)

with tabs[2]:
    ui["headaches"] = st.slider("Frekuensi Mengalami Sakit Kepala", 1, 5, 3)
    ui["health_issues"] = st.slider("Keluhan Sakit Fisik Lainnya", 1, 5, 3)
    ui["weight_changes"] = st.slider("Perubahan Berat Badan Secara Drastis", 1, 5, 3)

with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        ui["academic_overload"] = st.slider("Tekanan & Volume Tugas Kuliah", 1, 5, 3)
        ui["peer_competition"] = st.slider("Tekanan Kompetisi Antar Teman Sekelas", 1, 5, 3)
        ui["low_academic_confidence"] = st.slider("Rasa Kurang Percaya Diri terhadap Nilai", 1, 5, 3)
        ui["subject_confidence"] = st.slider("Tingkat Penguasaan Materi Kuliah", 1, 5, 3)
    with c2:
        ui["academic_conflicts"] = st.slider("Konflik Akademik (Dosen / Universitas)", 1, 5, 3)
        ui["class_attendance"] = st.slider("Tingkat Kehadiran Presensi Kelas", 1, 5, 3)
        ui["professor_difficulties"] = st.slider("Hambatan Komunikasi dengan Dosen", 1, 5, 3)
        ui["work_environment"] = st.slider("Kondisi Lingkungan Kerja / Tugas Kelompok", 1, 5, 3)
        ui["home_environment"] = st.slider("Kenyamanan Lingkungan Tempat Tinggal/Kos", 1, 5, 3)

with tabs[4]:
    ui["relationship_stress"] = st.slider("Stres Hubungan (Asmara / Keluarga)", 1, 5, 3)
    ui["lack_relaxation_time"] = st.slider("Kurangnya Waktu untuk Liburan/Relaksasi", 1, 5, 3)

    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.user_input = ui.copy()

    if st.button("Tampilkan Hasil Analisis", type="primary", use_container_width=True):
        st.session_state.is_analyzed = True

# === HASIL ANALISIS ===
    if st.session_state.is_analyzed:
        ui = st.session_state.user_input
        st.markdown("<br><hr style='border-color:#2C2C2E;'>", unsafe_allow_html=True)

        input_scaled = scaler.transform(pd.DataFrame([ui])[FEATURES])
        prediction   = model.predict(input_scaled)[0]

        st.markdown(
            f"<h2 style='text-align:center; color:{CLASS_COLORS[prediction]} !important;'>"
            f"Status: {CLASS_LABELS[prediction]}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(f"> {CLASS_DESC[prediction]}")
        st.markdown("<br>", unsafe_allow_html=True)

        # Rekomendasi berdasarkan nilai input user
        recommendations = []
        if ui["academic_overload"] >= 4 and ui["sleep_problems"] >= 4:
            recommendations.append("Beban belajar dan gangguan tidurmu berada di zona merah. Terapkan batasan tegas kapan harus berhenti belajar demi menjaga tubuh.")
        if ui["loneliness_isolation"] >= 4:
            recommendations.append("Interaksi sosialmu minim. Sempatkan menyapa teman lama atau ikut berdiskusi langsung di sekretariat/lingkungan kampus.")
        if ui["lack_relaxation_time"] >= 4:
            recommendations.append("Tubuhmu butuh rehat. Jadwalkan waktu kosong minimal 45 menit tanpa melihat notifikasi tugas sama sekali.")
        if ui["anxiety_tension"] >= 4 or ui["restlessness"] >= 4:
            recommendations.append("Tingkat kecemasan dan kegelisahanmu cukup tinggi. Teknik pernapasan dalam atau journaling singkat sebelum tidur bisa membantu meredakannya.")
        if not recommendations:
            recommendations.append("Kombinasi pola hidup dan aktivitas akademikmu sudah berada pada jalur yang sehat. Pertahankan keseimbangan ini!")

        st.markdown("#### Actionable Insights")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        st.markdown("<br>", unsafe_allow_html=True)

        # Visualisasi
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Faktor Pemicu Tertinggi**")
            top5 = sorted(
                {k: v for k, v in ui.items() if k not in ["gender", "age"]}.items(),
                key=lambda x: x[1], reverse=True
            )[:5]
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_facecolor("#1C1C1E")
            ax.set_facecolor("#1C1C1E")
            ax.barh([x[0].replace("_", " ").title() for x in top5], [x[1] for x in top5],
                    color="#0A84FF", height=0.55)
            ax.set_xlim(0, 5)
            ax.invert_yaxis()
            ax.tick_params(colors="#FFFFFF", labelsize=9)
            for spine in ["top", "right", "left"]:
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#2C2C2E")
            ax.grid(axis="x", color="#2C2C2E", linestyle="--", alpha=0.7)
            st.pyplot(fig)

        with col2:
            st.markdown("**Radar Keseimbangan Mental**")
            radar_features = ["sleep_problems", "academic_overload", "anxiety_tension",
                               "peer_competition", "relationship_stress"]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[ui[f] for f in radar_features],
                theta=[f.replace("_", " ").title() for f in radar_features],
                fill="toself", name="Skormu", line_color="#0A84FF"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[3.1, 3.4, 2.9, 3.0, 2.6],
                theta=[f.replace("_", " ").title() for f in radar_features],
                fill="toself", name="Rata-rata", line_color="rgba(142,142,147,0.4)"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5], gridcolor="#2C2C2E",
                                    tickfont=dict(color="#8E8E93")),
                    angularaxis=dict(gridcolor="#2C2C2E", tickfont=dict(color="#FFFFFF", size=10)),
                    bgcolor="rgba(0,0,0,0)"
                ),
                showlegend=True, legend=dict(font=dict(color="#FFFFFF")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=25, b=25)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Simulasi: slider arah berlawanan dengan skala form karena merepresentasikan tingkat perbaikan
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Simulasi Modifikasi Kebiasaan (Live Preview)", expanded=True):
            st.markdown(
                "<p style='color:#8E8E93; font-size:0.9em; margin-bottom:1.5rem;'>"
                "Geser slider di bawah ini untuk melihat simulasi perbaikan gaya hidupmu secara instan.</p>",
                unsafe_allow_html=True
            )
            sim_data = ui.copy()
            for feat in get_top_features(model, FEATURES):
                label, is_positive = FEATURE_META.get(feat, (feat.replace("_", " ").title(), False))
                if is_positive:
                    slider_label = f"Tingkatkan: {label} (1 = Sangat Rendah, 5 = Sangat Tinggi)"
                else:
                    slider_label = f"Perbaiki: {label} (1 = Masih Parah, 5 = Sudah Membaik)"
                val = st.slider(slider_label, 1, 5, 3, key=f"sim_{feat}")
                sim_data[feat] = val if is_positive else (6 - val)

            sim_pred = model.predict(scaler.transform(pd.DataFrame([sim_data])[FEATURES]))[0]
            st.markdown(
                f"<h4 style='text-align:center; margin-top:1rem;'>Prediksi Simulasi: "
                f"<span style='color:{CLASS_COLORS[sim_pred]}; font-weight:700;'>"
                f"{CLASS_LABELS[sim_pred]}</span></h4>",
                unsafe_allow_html=True
            )
            if sim_pred == 1:
                st.success("Hasil Bagus! Skenario perbaikan gaya hidup ini terbukti efektif menurunkan tingkat risiko stres menurut kalkulasi algoritma.")
            elif sim_pred == 2:
                st.info("Kondisi sudah bergerak ke arah lebih seimbang. Konsistensi perubahan kebiasaan ini akan mendorong hasil yang lebih stabil.")
            else:
                st.warning("Perubahan ini dirasa belum cukup dominan. Cobalah naikkan kualitas tidur atau kombinasikan dengan penyeimbang manajemen waktu lainnya.")

        # Download laporan
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#2C2C2E;'>", unsafe_allow_html=True)

        def build_report(ui, prediction, recommendations, top5):
            sep  = "=" * 52
            now  = datetime.now().strftime("%d %B %Y, %H:%M")
            top5_lines = "\n".join(
                f"  {i+1}. {k.replace('_',' ').title():<30}: {v}/5"
                for i, (k, v) in enumerate(top5)
            )
            detail_lines = []
            for kat, feats in KATEGORI_FEATURES.items():
                detail_lines.append(f"\n  [{kat}]")
                for f in feats:
                    label  = FEATURE_META.get(f, (f.replace("_", " ").title(), False))[0]
                    suffix = "/5" if f not in ["gender", "age"] else ""
                    detail_lines.append(f"\n  {label:<35}: {ui.get(f, '-')}{suffix}")
            rec_lines = "\n".join(f"  - {r}" for r in recommendations)
            return f"""{sep}
          LAPORAN ANALISIS STRES AKADEMIK
                  CampusCALM
{sep}
  Tanggal Analisis : {now}
{sep}

  HASIL PREDIKSI
  Status           : {CLASS_LABELS[prediction]}

  {CLASS_DESC[prediction].replace('**', '')}

{sep}

  TOP 5 FAKTOR PEMICU TERTINGGI
{top5_lines}

{sep}

  ACTIONABLE INSIGHTS
{rec_lines}

{sep}

  DETAIL DATA INPUT
{"".join(detail_lines)}

{sep}
  Laporan ini dihasilkan secara otomatis oleh CampusCALM.
  Bukan pengganti diagnosis profesional kesehatan mental.
{sep}""".strip()

        report_text = build_report(ui, prediction, recommendations, top5)
        st.download_button(
            label="Unduh Laporan Analisis (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"CampusCALM_Laporan_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )