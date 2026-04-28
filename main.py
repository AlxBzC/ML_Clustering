import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
from model import clasificar_madre

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(layout="wide", page_title="Análisis Madres - UAO")

# Colores consistentes para toda la interfaz
COLORES_CLUSTER = {
    'Cluster 0': "#a940ff",
    'Cluster 1': "#ffb30e",
    'Cluster 2': "#ff445d",
    'Cluster 3': "#34ff78"
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
    except:
        return None

# 2. FUNCIONES DE MAPAS
def generar_mapa_puntos(df, geojson_data):
    limites = {}
    for feature in geojson_data['features']:
        cod = feature['properties']['DPTO']
        coords = np.array(feature['geometry']['coordinates'][0])
        if coords.ndim == 3: coords = coords[0]
        min_lon, min_lat = coords.min(axis=0)
        max_lon, max_lat = coords.max(axis=0)
        
        # Margen de seguridad interno para evitar puntos en el mar o fuera
        m_lon = (max_lon - min_lon) * 0.15
        m_lat = (max_lat - min_lat) * 0.15
        limites[cod] = (min_lon + m_lon, max_lon - m_lon, min_lat + m_lat, max_lat - m_lat)

    puntos_lista = []
    for _, fila in df.iterrows():
        cod = fila['CODPTORE']
        if cod in limites:
            min_lon, max_lon, min_lat, max_lat = limites[cod]
            # Aseguramos que tome el color del diccionario usando el label correcto
            c_num = str(fila['Cluster']).replace('Cluster ', '')
            label = f"Cluster {c_num}"
            
            puntos_lista.append({
                'lat': np.random.uniform(min_lat, max_lat),
                'lon': np.random.uniform(min_lon, max_lon),
                'Cluster': label,
                'Departamento': NOMBRES_DEPTOS.get(cod, f"Dpto {cod}")
            })
    
    df_puntos = pd.DataFrame(puntos_lista)
    fig = px.scatter_mapbox(
        df_puntos, lat="lat", lon="lon", color="Cluster",
        color_discrete_map=COLORES_CLUSTER, 
        hover_name="Departamento",
        zoom=4.7, center={"lat": 4.5708, "lon": -74.2973},
        mapbox_style="carto-positron", height=700
    )
    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def generar_mapa_predominancia(df, geojson_data):
    df_local = df.copy()
    df_local['C_Num'] = df_local['Cluster'].astype(str).str.extract('(\d+)').fillna(0).astype(int)
    df_moda = df_local.groupby('CODPTORE')['C_Num'].agg(lambda x: x.mode()[0]).reset_index()
    df_moda['Cluster_Label'] = df_moda['C_Num'].apply(lambda x: f"Cluster {x}")
    df_moda['Nombre_Dpto'] = df_moda['CODPTORE'].apply(lambda x: NOMBRES_DEPTOS.get(x, "Desconocido"))

    fig = px.choropleth(
        df_moda, geojson=geojson_data, locations="CODPTORE",
        featureidkey="properties.DPTO", color="Cluster_Label",
        color_discrete_map=COLORES_CLUSTER, scope="south america",
        hover_name="Nombre_Dpto",
        hover_data={'CODPTORE': True, 'Cluster_Label': True, 'C_Num': False},
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
    return fig

# 3. INTERFAZ Y LÓGICA
st.title("📊 Dashboard Analítico de Perfiles Maternos")

archivo = st.file_uploader("Cargar Dataset (CSV)", type=["csv"])

if archivo is not None:
    df_nuevo = pd.read_csv(archivo, sep=';')

    # --- BLOQUE DE LIMPIEZA DE DATOS  ---
    with st.expander("🛠️ Preprocesamiento de datos"):
        st.write("Limpiando registros con valores 'Sin información' (99, 9, etc.)...")
        
        # Guardamos la cantidad original para informar cuánto se limpió
        registros_originales = len(df_nuevo)
        
        # Aplicamos tus filtros de limpieza
        df_nuevo = df_nuevo[
            (df_nuevo['PESO_NAC'] < 9) &
            (df_nuevo['T_GES'] < 9) &
            (df_nuevo['NUMCONSUL'] < 99) &
            (df_nuevo['TIPO_PARTO'] < 9) &
            (df_nuevo['APGAR1'] < 99) &
            (df_nuevo['IDPERTET'] < 9) &
            (df_nuevo['EDAD_MADRE'] < 99) &
            (df_nuevo['NIV_EDUM'] < 99) &
            (df_nuevo['AREA_RES'] < 9) &
            (df_nuevo['SEG_SOCIAL'] < 9) &
            (df_nuevo['EDAD_PADRE'] < 99) &
            (df_nuevo['NIV_EDUP'] < 99)
        ]
        
        limpiados = registros_originales - len(df_nuevo)
        st.info(f"Se eliminaron {limpiados} registros inconsistentes. Quedan {len(df_nuevo)} filas válidas.")
    
    with st.spinner('Ejecutando inteligencia de datos...'):
        resultados = clasificar_madre(df_nuevo)
    
    # Normalización de códigos
    resultados['CODPTORE'] = pd.to_numeric(resultados['CODPTORE'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(2)
    geojson = obtener_limites_geojson()

    # PESTAÑAS
    tab_resumen, tab_puntos, tab_regional = st.tabs(["📋 Resumen de Perfiles", "📍 Dispersión Geográfica", "🗺️ Mapa Regional"])

    with tab_resumen:
        st.subheader("Estadísticas Generales del Análisis")
        
        # Métricas principales
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Datos Procesados", f"{len(resultados):,}")
        m2.metric("Clusters Identificados", resultados['Cluster'].nunique())
        m3.metric("Departamento con más registros", NOMBRES_DEPTOS.get(resultados['CODPTORE'].mode()[0]))

        st.divider()
        
        st.subheader("Análisis de Características por Cluster")
        st.write("A continuación se presentan los promedios de las variables principales para cada perfil identificado:")
        
        # 1. Separamos las variables por tipo
        vars_categoricas = ['SEXO', 'TIPO_PARTO', 'AREA_RES', 'SEG_SOCIAL', 'IDPERTET', 'NIV_EDUM','NIV_EDUP']
        vars_numericas = ['EDAD_MADRE', 'EDAD_PADRE', 'NUMCONSUL', 'T_GES', 'PESO_NAC', 'APGAR1']

        # 2. Calculamos Medias para lo numérico y Modas para lo categórico
        resumen_num = resultados.groupby('Cluster')[vars_numericas].mean()
        resumen_cat = resultados.groupby('Cluster')[vars_categoricas].agg(lambda x: x.mode()[0])

         # Unimos ambos resultados en una sola tabla
        df_perfiles_pro = pd.concat([resumen_num, resumen_cat], axis=1).reset_index()

        st.write("### Perfil Promedio y Predominante")
        st.dataframe(df_perfiles_pro.style.format(precision=1).background_gradient(cmap='PuBu'), use_container_width=True)        
        
        # Gráfico comparativo rápido
        st.subheader("Distribución de la Población")
        fig_bar = px.histogram(resultados, x="Cluster", color="Cluster", 
                               color_discrete_map=COLORES_CLUSTER,
                               title="Cantidad de Madres por Cluster")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_puntos:
        st.plotly_chart(generar_mapa_puntos(resultados, geojson), use_container_width=True)

    with tab_regional:
        st.plotly_chart(generar_mapa_predominancia(resultados, geojson), use_container_width=True)