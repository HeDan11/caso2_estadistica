import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard – Caso 2", layout="wide")
st.title("Dashboard – Predicción del precio de propiedades (Caso 2)")

st.write("""
Este dashboard presenta los resultados del análisis estadístico y predictivo
realizado sobre una muestra de 50,000 propiedades del mercado inmobiliario de EE.UU.
Todas las gráficas y datos cargan automáticamente.
""")

# ======================
# Cargar archivos locales
# ======================

@st.cache_data
def cargar_dataset():
    return pd.read_csv("housing_sample_clean.csv")

housing = cargar_dataset()

# =======================
# Tabs
# =======================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Exploración", "📈 Correlación", "🤖 Modelos", "📝 Conclusiones"]
)

# ==========================================
# TAB 1: Exploración del dataset
# ==========================================
with tab1:
    st.header("Exploración del dataset")
    st.subheader("Vista previa del dataset limpio")
    st.dataframe(housing.head())

    st.subheader("Estadísticas descriptivas")
    st.dataframe(housing.describe())

    st.subheader("Distribución del precio de las propiedades")
    st.image("distribucion_precio.png", use_container_width=True)

# ==========================================
# TAB 2: Correlación
# ==========================================
with tab2:
    st.header("Matriz de correlación")
    st.write("""
La matriz de correlación muestra relaciones moderadas entre el precio y 
las variables estructurales como el número de recámaras, baños y tamaño de la vivienda.
    """)
    st.image("correlacion_housing.png", use_container_width=True)

# ==========================================
# TAB 3: Comparación de modelos
# ==========================================
with tab3:
    st.header("Modelos evaluados y métricas")

    resultados = pd.DataFrame({
        "Modelo": ["Regresión simple", "Regresión múltiple", "Random Forest", "Gradient Boosting"],
        "R2": [0.1683, 0.2662, 0.7220, 0.7091],
        "RMSE": [748029.43, 702626.18, 432480.11, 442408.82],
        "MAE": [425831.35, 304906.35, 199570.70, 230087.19]
    })

    st.subheader("Métricas obtenidas")
    st.dataframe(resultados.style.format({"R2": "{:.3f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}"}))

    st.subheader("Gráfica Real vs Predicho (Random Forest)")
    st.image("real_vs_pred_rf.png", use_container_width=True)

# ==========================================
# TAB 4: Conclusiones
# ==========================================
with tab4:
    st.header("Conclusiones del estudio")

    st.write("""
Los resultados del Caso 2 muestran que:

- La **regresión lineal simple y múltiple** presenta desempeño limitado  
  ($R^2$ entre 0.17 y 0.27), por lo que no captura la complejidad del mercado inmobiliario.

- Los modelos de *machine learning* basados en árboles (**Random Forest** y **Gradient Boosting**) 
  ofrecen un rendimiento muy superior ($R^2 \approx 0.72$).

- El **Random Forest** fue el mejor modelo del estudio.

### Nota importante  
Debido a limitaciones de tiempo y cómputo, **no se realizó tuning de hiperparámetros**.  
Se espera que una búsqueda sistemática (GridSearch/RandomSearch) mejore significativamente el desempeño obtenido.

### Recomendaciones futuras
- Agregar variables como antigüedad, coordenadas geográficas, calidad del vecindario, etc.
- Realizar tuning de hiperparámetros.
- Construir un pipeline completo para predicción inmobiliaria.
""")


