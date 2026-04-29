import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import os
from model import clasificar_madre 

# 1. CONFIGURACIÓN VISUAL
st.set_page_config(
    layout="wide", 
    page_title="Perfiles de Madres",
    page_icon="📊"
)

# Paleta de colores profesionales
COLORES_PERFIL = {
    'Perfil 0': "#F5516C",
    'Perfil 1': "#0B2F9A",
    'Perfil 2': "#EA8B6C",
    'Perfil 3': "#72DDF0"
}

NOMBRES_DEPTOS = {
    '05': 'Antioquia', '08': 'Atlántico', '11': 'Bogotá, D.C.', '13': 'Bolívar', 
    '15': 'Boyacá', '17': 'Caldas', '18': 'Caquetá', '19': 'Cauca', '20': 'Cesar', 
    '23': 'Córdoba', '25': 'Cundinamarca', '27': 'Chocó', '41': 'Huila', 
    '44': 'La Guajira', '47': 'Magdalena', '50': 'Meta', '52': 'Nariño', 
    '54': 'Norte de Santander', '63': 'Quindío', '66': 'Risaralda', 
    '68': 'Santander', '70': 'Sucre', '73': 'Tolima', '76': 'Valle del Cauca', 
    '81': 'Arauca', '85': 'Casanare', '86': 'Putumayo', 
    '88': 'San Andrés y Providencia', '91': 'Amazonas', '94': 'Guainía', 
    '95': 'Guaviare', '97': 'Vaupés', '99': 'Vichada'
}

@st.cache_data
def obtener_limites_geojson():
    url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
    try:
        resp = requests.get(url)
        return resp.json()
    except: return None

# --- LÓGICA DE MAPAS ORIGINAL (SIN CAMBIOS) ---
def generar_mapa_puntos(df, geojson_data):
    limites = {}
    for feature in geojson_data['features']:
        cod = feature['properties']['DPTO']
        coords = np.array(feature['geometry']['coordinates'][0])
        if coords.ndim == 3: coords = coords[0]
        min_lon, min_lat = coords.min(axis=0)
        max_lon, max_lat = coords.max(axis=0)
        m_lon, m_lat = (max_lon - min_lon) * 0.15, (max_lat - min_lat) * 0.15
        limites[cod] = (min_lon + m_lon, max_lon - m_lon, min_lat + m_lat, max_lat - m_lat)

    puntos_lista = []
    for _, fila in df.iterrows():
        cod = fila['CODPTORE']
        if cod in limites:
            min_lon, max_lon, min_lat, max_lat = limites[cod]
            c_num = str(fila['Cluster']).replace('Cluster ', '')
            puntos_lista.append({
                'lat': np.random.uniform(min_lat, max_lat),
                'lon': np.random.uniform(min_lon, max_lon),
                'Perfil': f"Perfil {c_num}",
                'Departamento': NOMBRES_DEPTOS.get(cod, f"Dpto {cod}")
            })
    
    df_puntos = pd.DataFrame(puntos_lista)
    fig = px.scatter_mapbox(
        df_puntos, lat="lat", lon="lon", color="Perfil",
        color_discrete_map=COLORES_PERFIL, hover_name="Departamento",
        zoom=4.7, center={"lat": 4.5708, "lon": -74.2973},
        mapbox_style="carto-positron", height=700
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

def generar_mapa_predominancia(df, geojson_data):
    df_local = df.copy()
    df_local['C_Num'] = df_local['Cluster'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
    df_moda = df_local.groupby('CODPTORE')['C_Num'].agg(lambda x: x.mode()[0]).reset_index()
    df_moda['Perfiles'] = df_moda['C_Num'].apply(lambda x: f"Perfil {x}")
    df_moda['Nombre_Dpto'] = df_moda['CODPTORE'].apply(lambda x: NOMBRES_DEPTOS.get(x, "Desconocido"))

    fig = px.choropleth(
        df_moda, geojson=geojson_data, locations="CODPTORE",
        featureidkey="properties.DPTO", color="Perfiles",
        color_discrete_map=COLORES_PERFIL, scope="south america",
        hover_name="Nombre_Dpto",
        hover_data={'CODPTORE': True, 'Perfiles': True, 'C_Num': False},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
    fig.update_traces(marker_line_color="white", marker_line_width=1.0)
    return fig

# --- CABECERA ---
st.title("Identificación de perfiles y condiciones de vida de las madres y la salud neonatal en Colombia")

ruta_imagen = "imagen.png"
if os.path.exists(ruta_imagen):
    st.image(ruta_imagen, use_container_width=True)

st.markdown("---")

# BARRA LATERAL
with st.sidebar:
    st.header("⚙️ Configuración")
    archivo = st.file_uploader("Cargar dataset de nacimientos (CSV)", type=["csv"])

if archivo is not None:
    df_nuevo = pd.read_csv(archivo, sep=';')
    
    # Preprocesamiento original
    df_nuevo = df_nuevo[
        (df_nuevo['PESO_NAC'] < 9) & (df_nuevo['T_GES'] < 9) & (df_nuevo['NUMCONSUL'] < 99) &
        (df_nuevo['TIPO_PARTO'] < 9) & (df_nuevo['APGAR1'] < 99) & (df_nuevo['IDPERTET'] < 9) &
        (df_nuevo['EDAD_MADRE'] < 99) & (df_nuevo['NIV_EDUM'] < 99) & (df_nuevo['AREA_RES'] < 9) &
        (df_nuevo['SEG_SOCIAL'] < 9) & (df_nuevo['EDAD_PADRE'] < 99) & (df_nuevo['NIV_EDUP'] < 99)
    ]

    with st.spinner('Analizando datos...'):
        resultados = clasificar_madre(df_nuevo)
    
    resultados['CODPTORE'] = pd.to_numeric(resultados['CODPTORE'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(2)
    geojson = obtener_limites_geojson()

    tab_resumen, tab_puntos, tab_regional = st.tabs(["📊 Resumen Perfiles", "📍 Mapa Dispersión", "🗺️ Mapa Regional"])

    with tab_resumen:
        c1, c2, c3 = st.columns(3)
        c1.metric("Procesados", f"{len(resultados):,}")
        c2.metric("Perfiles", resultados['Cluster'].nunique())
        c3.metric("Dpto Principal", NOMBRES_DEPTOS.get(resultados['CODPTORE'].mode()[0]))

        st.subheader("Caracterización por Perfil")
        vars_cat = ['SEXO', 'TIPO_PARTO', 'AREA_RES', 'SEG_SOCIAL', 'IDPERTET', 'NIV_EDUM']
        vars_num = ['EDAD_MADRE', 'EDAD_PADRE', 'NUMCONSUL', 'T_GES', 'PESO_NAC']
        
        # DICCIONARIO DE VARIABLES ---
        st.markdown("---")
        st.subheader("📋 Glosario de Variables y Códigos")
        st.info("Utilice las secciones desplegables a continuación para entender el significado de los códigos numéricos en los perfiles identificados.")

        col_var1, col_var2 = st.columns(2)

        with col_var1:
            with st.expander("Información de la Madre"):
                st.markdown("**EDAD_MADRE (Años)**")
                st.caption("**1:** 10-14  | **2:** 15-19  | **3:** 20-24  | **4:** 25-29  | **5:** 30-34  | **6:** 35-39  | **7:** 40-44  | **8:** 45-49  | **9:** 50-54 ")
                
                st.markdown("**NIV_EDUM (Nivel Educativo)**")
                st.caption("**1:** Preescolar | **2:** Primaria | **3:** Secundaria | **4:** Media | **5:** M. Técnica | **6:** Normalista | **7:** T. Profesional | **8:** Tecnológica | **9:** Profesional | **10:** Especialización | **11:** Maestría | **12:** Doctorado | **13:** Ninguno")
                
                st.markdown("**SEG_SOCIAL**")
                st.caption("**1:** Contributivo | **2:** Subsidiado | **3:** Excepción | **4:** Especial | **5:** No asegurado")
                
                st.markdown("**AREA_RES**")
                st.caption("**1:** Cabecera municipal | **2:** Centro poblado | **3:** Rural disperso")

            with st.expander("Detalles del Nacido Vivo"):
                st.markdown("**SEXO**")
                st.caption("**1:** Masculino | **2:** Femenino")
                
                st.markdown("**PESO_NAC (Gramos)**")
                st.caption("**1:** <1.000 | **2:** 1.000-1.499 | **3:** 1.500-1.999 | **4:** 2.000-2.499 | **5:** 2.500-2.999 | **6:** 3.000-3.499 | **7:** 3.500-3.999 | **8:** >4.000")
                
                st.markdown("**T_GES (Semanas)**")
                st.caption("**1:** <22 | **2:** 22-27 | **3:** 28-37 | **4:** 38-41 | **5:** >42 | **6:** Ignorado")
                
                st.markdown("**IDPERTET (Etnia)**")
                st.caption("**1:** Indígena | **2:** Rom | **3:** Raizal | **4:** Palenquero | **5:** Afro / Mulato | **6:** Ninguna")

        with col_var2:
            with st.expander("Proceso de Parto y Salud"):
                st.markdown("**NUMCONSUL**")
                st.caption("Representa el número total de consultas prenatales reportadas.")
                
                st.markdown("**TIPO_PARTO**")
                st.caption("**1:** Espontáneo | **2:** Cesárea | **3:** Instrumentado | **4:** Ignorado")

            with st.expander("Información del Padre"):
                st.markdown("**EDAD_PADRE**")
                st.caption("Edad en años cumplidos reportada en el certificado.")
                
                st.markdown("**NIV_EDUP (Nivel Educativo)**")
                st.caption("**1:** Preescolar | **2:** Primaria | **3:** Secundaria | **4:** Media | **5:** M Técnica | **6:** Normalista | **7:** T Profesional | **8:** Tecnológica | **9:** Profesional | **10:** Especialización | **11:** Maestría | **12:** Doctorado | **13:** Ninguno")

        st.markdown("---")
        #####
        res_num = resultados.groupby('Cluster')[vars_num].mean()
        res_cat = resultados.groupby('Cluster')[vars_cat].agg(lambda x: x.mode()[0])
        df_pro = pd.concat([res_num, res_cat], axis=1).reset_index()
        df_pro['Cluster'] = df_pro['Cluster'].astype(str).replace('Cluster ', 'Perfil ', regex=True)
        df_pro.rename(columns={'Cluster': 'Perfil'}, inplace=True)
        
        st.dataframe(df_pro.style.format(precision=1).background_gradient(cmap='Blues', subset=vars_num), use_container_width=True)

        # Histograma 
        dist = resultados['Cluster'].value_counts().reset_index()
        dist.columns = ['Perfil', 'Cantidad']
        dist['Perfil'] = dist['Perfil'].astype(str).replace('Cluster ', 'Perfil ', regex=True)
        st.plotly_chart(px.bar(dist, x='Perfil', y='Cantidad', color='Perfil', color_discrete_map=COLORES_PERFIL), use_container_width=True)

    with tab_puntos:
        st.markdown("Esta visualización cartográfica emplea un algoritmo de distribución estocástica " \
        "para representar la densidad poblacional de los perfiles identificados a nivel departamental, " \
        "facilitando el análisis de micro-focos de vulnerabilidad o características específicas" \
        " de los diferentes perfiles maternos.")
        st.plotly_chart(generar_mapa_puntos(resultados, geojson), use_container_width=True)

    with tab_regional:
        st.markdown("Este mapa visualiza la variable de tendencia central (moda estadística) " \
        "de los perfiles maternos por departamento")
        st.plotly_chart(generar_mapa_predominancia(resultados, geojson), use_container_width=True)
else:
    st.info("Cargue un archivo CSV para comenzar.")
