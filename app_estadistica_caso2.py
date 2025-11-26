import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Dashboard – Predicción de precios de propiedades",
    layout="wide"
)

st.title("Dashboard – Caso 2: Predicción del precio de propiedades")
st.write("""
Este dashboard presenta de forma resumida los resultados del análisis estadístico y de 
modelado predictivo realizado sobre un dataset de bienes raíces de Estados Unidos.
Incluye exploración de datos, análisis de correlación y comparación de modelos de regresión
(lineales y de *machine learning*).
""")

st.markdown("---")

# =========================
# SIDEBAR: ARCHIVOS
# =========================
st.sidebar.header("Archivos de entrada")

data_file = st.sidebar.file_uploader(
    "Sube el CSV del dataset limpio (housing_sample_clean.csv)",
    type="csv"
)

corr_img = st.sidebar.file_uploader(
    "Sube la imagen de la matriz de correlación (correlacion_housing.png)",
    type=["png", "jpg", "jpeg"]
)

rvp_img = st.sidebar.file_uploader(
    "Sube la imagen Real vs Predicho (real_vs_pred_rf.png)",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.markdown("---")
st.sidebar.write("Las métricas de los modelos se cargan desde los resultados obtenidos en Colab.")

# =========================
# TABS PRINCIPALES
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Exploración de datos", "📈 Correlación", "🤖 Modelos y métricas", "📝 Conclusiones"]
)

# =========================
# TAB 1: EXPLORACIÓN
# =========================
with tab1:
    st.header("Exploración de datos")

    if data_file is not None:
        housing = pd.read_csv(data_file)

        st.subheader("Vista previa del dataset")
        st.dataframe(housing.head())

        st.subheader("Estadísticas descriptivas")
        st.dataframe(housing.describe())

        st.subheader("Información general")
        col1, col2, col3 = st.columns(3)
        col1.metric("Número de registros", f"{housing.shape[0]:,}")
        col2.metric("Número de variables", f"{housing.shape[1]:,}")
        col3.metric("Variable objetivo", "price")

    else:
        st.info("Sube el archivo CSV en la barra lateral para ver la exploración de datos.")

# =========================
# TAB 2: CORRELACIÓN
# =========================
with tab2:
    st.header("Análisis de correlación")

    st.write("""
La matriz de correlación permite identificar qué variables numéricas muestran mayor
relación lineal con el precio de la propiedad. En particular, se observan asociaciones
moderadas entre el precio y el número de baños, recámaras y el tamaño de la vivienda.
    """)

    if corr_img is not None:
        st.image(corr_img, caption="Matriz de correlación de variables numéricas", use_container_width=True)
    else:
        st.info("Sube la imagen de la matriz de correlación en la barra lateral.")

# =========================
# TAB 3: MODELOS
# =========================
with tab3:
    st.header("Comparación de modelos de regresión")

    st.write("""
Se evaluaron cuatro modelos de regresión:

- **Regresión lineal simple** (price ~ house_size)  
- **Regresión lineal múltiple** (variables estructurales y de ubicación codificada)  
- **Random Forest Regressor**  
- **Gradient Boosting Regressor**

Las métricas mostradas a continuación fueron calculadas en Colab sobre el conjunto de prueba.
    """)

    # Tabla de resultados con tus métricas reales
    resultados = pd.DataFrame({
        "Modelo": ["Regresión simple", "Regresión múltiple", "Random Forest", "Gradient Boosting"],
        "R2": [0.1683, 0.2662, 0.7220, 0.7091],
        "RMSE": [748029.43, 702626.18, 432480.11, 442408.82],
        "MAE": [425831.35, 304906.35, 199570.70, 230087.19]
    })

    st.subheader("Métricas de desempeño")
    st.dataframe(resultados.style.format({"R2": "{:.3f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}"}))

    if rvp_img is not None:
        st.subheader("Gráfica Real vs Predicho (Random Forest)")
        st.image(rvp_img, caption="Relación entre precios reales y predichos", use_container_width=True)
    else:
        st.info("Sube la imagen Real vs Predicho en la barra lateral para visualizarla aquí.")

# =========================
# TAB 4: CONCLUSIONES
# =========================
with tab4:
    st.header("Conclusiones del Caso 2")

    st.write("""
Los resultados muestran que:

- La **regresión lineal simple** y la **regresión múltiple** presentan un desempeño limitado \
con valores de $R^2$ cercanos a 0.17 y 0.27, respectivamente. Esto indica que los modelos lineales \
no son capaces de capturar la complejidad del mercado inmobiliario con las variables disponibles.

- Los modelos de *machine learning* basados en árboles, **Random Forest** y **Gradient Boosting**, \
mejoran significativamente el ajuste, alcanzando $R^2$ del orden de 0.72 y reduciendo de manera importante \
las métricas de error (RMSE y MAE).

- El **Random Forest** fue el modelo con mejor desempeño global, por lo que se considera el candidato \
más adecuado para una implementación posterior de un sistema de predicción de precios.

Debido a las limitaciones de tiempo y recursos computacionales, **no se realizó una búsqueda exhaustiva \
de hiperparámetros ni una regularización avanzada**. Se espera que un proceso de *tuning* sistemático \
(profundidad máxima de los árboles, número de estimadores, tasas de aprendizaje, etc.) pueda mejorar \
aún más el desempeño obtenido.

En trabajos futuros se recomienda:
- Incorporar más variables relevantes (antigüedad de la propiedad, coordenadas geográficas, \
indicadores socioeconómicos del vecindario, calidad de construcción, entre otros).
- Aplicar técnicas de selección de características y reducción de dimensionalidad.
- Implementar un pipeline completo que integre el entrenamiento, validación y despliegue del modelo \
dentro de una aplicación web.
    """)
