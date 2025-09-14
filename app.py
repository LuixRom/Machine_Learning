# app_gpa.py
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =======================
# Configuración y logging
# =======================
st.set_page_config(page_title="GPA Early Support", layout="wide")

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info("App started")

# =======================
# Cargar pipeline
# =======================
@st.cache_resource
def load_pipeline():
    model_path = "student_gpa_predictor.pkl"  # <-- guarda tu Pipeline con este nombre
    pipe = joblib.load(model_path)            # Pipeline(prep + ElasticNetCV)
    return pipe

try:
    pipe = load_pipeline()
    st.success("✅ Modelo (pipeline) cargado correctamente")
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo: {e}")
    st.stop()

# =======================
# Utilidades
# =======================
# Define aquí EXACTAMENTE las columnas que tu pipeline espera (train_cols)
NUM_COLS = ["Age", "StudyTimeWeekly", "Absences"]
CAT_COLS = ["ParentalEducation", "Tutoring", "ParentalSupport", "Extracurricular",
            "Sports", "Music", "Volunteering"]
# Sensibles (para evaluación, NO para predicción)
SENSITIVE = ["Gender", "Ethnicity"]

TRAIN_COLS = NUM_COLS + CAT_COLS  # tu script así lo definía (sin sensibles ni ID/target)

def risk_bucket(gpa_pred: float) -> str:
    # Ajusta umbrales a tu escala (ej. 0-4). Ejemplo:
    if gpa_pred < 2.5:
        return "⚠️ Refuerzo Prioritario"
    elif gpa_pred < 3.0:
        return "🔶 Refuerzo Sugerido"
    else:
        return "✅ Progreso Adecuado"

def recommendations(row: pd.Series, base_pred: float) -> list:
    """
    Reglas simples y accionables basadas en features.
    (Puedes sofisticar esto con efectos marginales o SHAP)
    """
    recs = []
    # Heurísticas básicas:
    if row.get("StudyTimeWeekly", 0) < 8:
        recs.append("Incrementa tu estudio en +2 h/semana y divide en bloques cortos (Pomodoro).")
    if row.get("Absences", 0) > 3:
        recs.append("Reduce ausencias; prioriza asistencia las semanas con evaluaciones.")
    if row.get("Tutoring") == "No":
        recs.append("Inscríbete en tutorías gratuitas esta semana (30–45 min por curso crítico).")
    if row.get("ParentalSupport") in ["Low", "None"]:
        recs.append("Busca apoyo de pares/mentor académico para estructurar tu plan semanal.")
    if row.get("Extracurricular") == "No":
        recs.append("Considera una actividad breve que te motive (club/arte/deporte) para balance.")
    if not recs:
        recs.append("Mantén tu ritmo y calendariza revisiones semanales de progreso.")
    return recs

def simulate_delta(pipe, row_df, feature, delta):
    """
    Estima beneficio marginal simple: predicción(row) vs predicción(row con cambio).
    Útil para mensajes tipo 'si aumentas +2h estudio ≈ +ΔGPA'.
    """
    row2 = row_df.copy()
    if feature in row2.columns:
        row2.iloc[0, row2.columns.get_loc(feature)] = row2[feature].iloc[0] + delta
    try:
        p1 = float(pipe.predict(row_df)[0])
        p2 = float(pipe.predict(row2)[0])
        return p2 - p1
    except Exception:
        return np.nan

def group_fairness_metrics(y_true, y_pred, df_sensitive, group_col):
    """
    MAE/RMSE por grupo (NO usado para predecir; solo evaluación).
    """
    out = []
    for g, sub in df_sensitive.groupby(group_col):
        idx = sub.index
        e = y_true.loc[idx] - y_pred.loc[idx]
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e**2))
        out.append({"group": f"{group_col}={g}", "n": len(idx), "MAE": mae, "RMSE": rmse})
    return pd.DataFrame(out).sort_values("MAE")

# =======================
# Layout con pestañas
# =======================
tab1, tab2 = st.tabs(["👩‍🎓 Estudiante", "🏫 Coordinador/a"])

# ---------------
# Pestaña Estudiante
# ---------------
with tab1:
    st.header("Predicción de GPA y plan de mejora")
    col1, col2, col3 = st.columns([1,1,1])

    # Inputs NUMÉRICOS
    with col1:
        Age = st.number_input("Edad", min_value=15, max_value=80, value=18, step=1)
        StudyTimeWeekly = st.number_input("Horas de estudio/semana", min_value=0.0, value=6.0, step=0.5)
        Absences = st.number_input("Faltas (ciclo actual)", min_value=0, value=2, step=1)

    # Inputs CATEGÓRICOS
    with col2:
        ParentalEducation = st.selectbox("Educación de padres", ["None","Primary","Secondary","Higher"])
        Tutoring = st.selectbox("¿Asiste a tutorías?", ["No","Yes"])
        ParentalSupport = st.selectbox("Apoyo de padres", ["None","Low","Medium","High"])

    with col3:
        Extracurricular = st.selectbox("Actividades extracurriculares", ["No","Yes"])
        Sports = st.selectbox("Deportes", ["No","Yes"])
        Music = st.selectbox("Música", ["No","Yes"])
        Volunteering = st.selectbox("Voluntariado", ["No","Yes"])

    # (Opcional) Para evaluación, NO para predecir
    st.markdown("—")
    st.caption("⚖️ Los siguientes campos NO se usan para predecir. Solo ayudan a evaluar equidad en la pestaña de coordinador.")
    c1, c2 = st.columns(2)
    with c1:
        Gender = st.selectbox("Género (opcional)", ["Prefiero no decir", "Female", "Male", "Other"])
    with c2:
        Ethnicity = st.selectbox("Etnia (opcional)", ["Prefiero no decir", "GroupA", "GroupB", "GroupC"])

    if st.button("Calcular predicción", type="primary", use_container_width=True):
        start = datetime.now()
        # Armar DataFrame EXACTO con TRAIN_COLS
        row = {
            "Age": Age,
            "StudyTimeWeekly": StudyTimeWeekly,
            "Absences": Absences,
            "ParentalEducation": ParentalEducation,
            "Tutoring": Tutoring,
            "ParentalSupport": ParentalSupport,
            "Extracurricular": Extracurricular,
            "Sports": Sports,
            "Music": Music,
            "Volunteering": Volunteering,
        }
        X_row = pd.DataFrame([row], columns=TRAIN_COLS)

        try:
            gpa_pred = float(pipe.predict(X_row)[0])
            end = datetime.now()
            latency = (end - start).total_seconds()
            logger.info(f"Prediction made: {gpa_pred:.3f} in {latency:.3f}s")

            # Puedes estimar un rango con RMSE del test (si lo guardas). Aquí fijo ±0.30 como ejemplo.
            approx_rmse = 0.30
            st.subheader(f"Tu GPA estimado: **{gpa_pred:.2f}**")
            st.caption(f"Rango aproximado: {gpa_pred-approx_rmse:.2f} – {gpa_pred+approx_rmse:.2f}")

            st.markdown(f"**Estado:** {risk_bucket(gpa_pred)}")

            # Recomendaciones accionables
            recs = recommendations(pd.Series(row), gpa_pred)
            st.markdown("### Recomendaciones accionables")
            for r in recs:
                st.write(f"• {r}")

            # Mini-simulación de impacto de +2h/semana
            delta = simulate_delta(pipe, X_row, "StudyTimeWeekly", +2)
            if not np.isnan(delta):
                st.info(f"Si estudias **+2 h/semana**, tu GPA podría subir ~ **{delta:+.2f}** (estimación).")

        except Exception as e:
            st.error(f"Error en predicción: {e}")

# ---------------
# Pestaña Coordinador/a
# ---------------
with tab2:
    st.header("Monitoreo de riesgo y equidad")
    st.write("Sube un CSV con columnas **train_cols** + `GPA` (real) y opcionalmente `Gender`, `Ethnicity` para análisis de fairness.")
    up = st.file_uploader("Archivo CSV", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(up)
            # Validación mínima
            missing = [c for c in TRAIN_COLS if c not in data.columns]
            if missing:
                st.error(f"Faltan columnas requeridas: {missing}")
            elif "GPA" not in data.columns:
                st.error("Falta la columna objetivo 'GPA'")
            else:
                # Predicción
                X = data[TRAIN_COLS].copy()
                y = data["GPA"].astype(float)
                y_hat = pd.Series(pipe.predict(X), index=data.index, name="y_hat")

                # Métricas globales
                err = y - y_hat
                mae = np.mean(np.abs(err))
                rmse = np.sqrt(np.mean(err**2))
                r2 = 1 - (np.sum(err**2)/np.sum((y - y.mean())**2))

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.3f}")
                c2.metric("RMSE", f"{rmse:.3f}")
                c3.metric("R²", f"{r2:.3f}")

                # Ranking de casos prioritarios
                st.markdown("### Casos a priorizar (mayor error negativo: predicción optimista)")
                data_view = data.copy()
                data_view["y_hat"] = y_hat
                data_view["error"] = (y - y_hat)
                st.dataframe(
                    data_view.sort_values("error").head(10)[
                        ["StudentID", "GPA", "y_hat", "error", "StudyTimeWeekly", "Absences", "Tutoring"]
                        if "StudentID" in data_view.columns else
                        ["GPA", "y_hat", "error", "StudyTimeWeekly", "Absences", "Tutoring"]
                    ],
                    use_container_width=True
                )

                # Fairness por grupo (si hay columnas)
                st.markdown("### Fairness por grupo")
                if "Gender" in data.columns:
                    df_g = group_fairness_metrics(y, y_hat, data[["Gender"]], "Gender")
                    st.subheader("Por Género")
                    st.dataframe(df_g, use_container_width=True)
                else:
                    st.caption("No se encontró columna 'Gender'.")

                if "Ethnicity" in data.columns:
                    df_e = group_fairness_metrics(y, y_hat, data[["Ethnicity"]], "Ethnicity")
                    st.subheader("Por Etnia")
                    st.dataframe(df_e, use_container_width=True)
                else:
                    st.caption("No se encontró columna 'Ethnicity'.")

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
