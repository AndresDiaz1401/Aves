import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pandas as pd
import os
import urllib.parse
import gdown

# Lista de nombres de las especies
especiesAves = [
    'Capuchino Carinegro', 'Capuchino Tricolor', 'Mirla Cafe', 'Mirla Ojiblanco',
    'Mirlo Grande', 'Mirlo Serrano Andino', 'Mirlo sp', 'Zorzal Piquinegro',
    'Zorzal Ventricastaño', 'Zorzal_Mirlo sp'
]

# Cargar el modelo
# ID de tu archivo de Google Drive
file_id = "1hdMoII1lDavBKthmgiOcZn_vO1uZL_ca"
# Enlace directo (convertido)
url = f"https://drive.google.com/uc?id={file_id}"

# Nombre temporal del archivo descargado
model_path = "best_model.keras"

# Descargar el modelo desde Drive
gdown.download(url, model_path, quiet=False)

# Cargar el modelo descargado
modelo = load_model(model_path)
print("✅ Modelo cargado correctamente desde Google Drive")


# Cargar la base de datos desde Excel
@st.cache_data
def cargar_bd_excel():
    bd = pd.read_excel("db aves2.xlsx")
    bd["Nombre comun"] = bd["Nombre comun"].astype(str).str.strip().str.lower()
    return bd

# Función para cargar y preprocesar la imagen
def cargarProcesarImagenes(rutaImagen, tamaño=(224, 224)):
    imgen = image.load_img(rutaImagen, target_size=tamaño)
    arregloImagen = image.img_to_array(imgen)
    arregloImagen = np.expand_dims(arregloImagen, axis=0)
    arregloImagen = preprocess_input(arregloImagen)
    return arregloImagen

# Interfaz
st.title("Modelo de aves - Andrés Díaz")
st.write("Selecciona una imagen del ave:")

# Subir imagen
imagenCargada = st.file_uploader("Cargar imagen", type=["jpg", "jpeg", "png"])

if imagenCargada is not None:
    imagenTemporal = 'imagenTemporal.jpg'
    with open(imagenTemporal, "wb") as f:
        f.write(imagenCargada.getvalue())

    imagenArreglo = cargarProcesarImagenes(imagenTemporal)
    prediccion = modelo.predict(imagenArreglo)

    # Índices ordenados por probabilidad (descendente)
    top_indices = np.argsort(prediccion[0])[::-1]
    clasePredicha = top_indices[0]
    especiePrincipal = especiesAves[clasePredicha]
    probabilidadPrincipal = prediccion[0][clasePredicha]

    st.success(f"Especie predicha: {especiePrincipal}")
    st.info(f"Precisión del modelo: {probabilidadPrincipal * 100:.2f}%")

    # Mostrar imagen
    img_cv = cv2.imread(imagenTemporal)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title(f"{especiePrincipal} ({probabilidadPrincipal * 100:.2f}%)")
    ax.axis('off')
    st.pyplot(fig)
    os.remove(imagenTemporal)

    # Buscar información en base de datos
    bd = cargar_bd_excel()
    especieNormalizada = especiePrincipal.strip().lower()
    info_especie = bd[bd["Nombre comun"] == especieNormalizada]

    st.subheader("Información de la especie")
    if not info_especie.empty:
        for index, row in info_especie.iterrows():
            st.markdown(
                f"""
                <style>
                    .custom-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                        background-color: #f9f9f9;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                    .custom-table th, .custom-table td {{
                        padding: 12px 15px;
                        text-align: left;
                        font-size: 16px;
                    }}
                    .custom-table th {{
                        background-color: #2c3e50;
                        color: white;
                        width: 30%;
                    }}
                    .custom-table td {{
                        background-color: #808080;
                        color: white;
                    }}
                </style>
                <table class="custom-table">
                    <tr><th>Nombre comun:</th><td>{row['Nombre comun'].title()}</td></tr>
                    <tr><th>Nombre cientifico:</th><td>{row['Nombre cientifico']}</td></tr>
                    <tr><th>Color:</th><td>{row['Color']}</td></tr>
                    <tr><th>Alimentacion:</th><td>{row['Alimentacion']}</td></tr>
                    <tr><th>Comportamiento:</th><td>{row['Comportamiento']}</td></tr>
                    <tr><th>Conservación:</th><td>{row['Conservacion']}</td></tr>
                </table>
                """,
                unsafe_allow_html=True
            )
            busqueda_google = urllib.parse.quote(f"{row['Nombre comun']} ave")
            url_google = f"https://www.google.com/search?q={busqueda_google}"
            st.markdown(f"""
                <a href="{url_google}" target="_blank">
                    <button style='padding: 10px; font-size: 16px; border-radius: 8px; background-color: #4285F4; color: white; border: none; cursor: pointer;'>
                        🔎 Buscar en Google
                    </button>
                </a>
            """, unsafe_allow_html=True)

    # Mostrar otras especies alternativas (sin incluir la principal)
    st.subheader("¿No es la especie que esperas?")
    mostradas = 0
    for idx in top_indices[1:]:
        especieAlt = especiesAves[idx]
        if especieAlt == especiePrincipal:
            continue  # Evitar mostrar la especie principal de nuevo

        probabilidadAlt = prediccion[0][idx]
        especieAltNorm = especieAlt.strip().lower()
        info_especie = bd[bd["Nombre comun"] == especieAltNorm]

        with st.expander(f"{especieAlt} ({probabilidadAlt * 100:.2f}%)"):
            if not info_especie.empty:
                for _, row in info_especie.iterrows():
                    st.markdown(
                        f"""
                        <table class="custom-table">
                            <tr><th>Nombre comun:</th><td>{row['Nombre comun'].title()}</td></tr>
                            <tr><th>Nombre cientifico:</th><td>{row['Nombre cientifico']}</td></tr>
                            <tr><th>Color:</th><td>{row['Color']}</td></tr>
                            <tr><th>Alimentacion:</th><td>{row['Alimentacion']}</td></tr>
                            <tr><th>Comportamiento:</th><td>{row['Comportamiento']}</td></tr>
                            <tr><th>Conservación:</th><td>{row['Conservacion']}</td></tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )

            busqueda_google = urllib.parse.quote(f"{especieAlt} ave")
            url_google = f"https://www.google.com/search?q={busqueda_google}"
            st.markdown(f"""
                <a href="{url_google}" target="_blank">
                    <button style='margin-top:20px; padding: 10px 20px; font-size: 16px; border-radius: 8px; background-color: #4285F4; color: white; border: none; cursor: pointer;'>
                        Buscar en Google
                    </button>
                </a>
            """, unsafe_allow_html=True)

            mostradas += 1
            if mostradas == 3:
                break
