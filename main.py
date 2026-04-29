# ================================
# IMPORTACIÓN DE LIBRERÍAS
# ================================
# Se importan todas las librerías necesarias para:
# - Interfaz web (Streamlit)
# - Manipulación de datos (Pandas, NumPy)
# - Visualización (Plotly)
# - Consumo de servicios (Requests)
# - Manejo de archivos (os, base64)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import os
import base64

# Importación del modelo de Machine Learning
from model import clasificar_madre 

# ================================
# CONFIGURACIÓN GENERAL DE LA APP
# ================================
# Define el diseño, título y el ícono de la aplicación

st.set_page_config(
    layout="wide", 
    page_title="Perfiles de Madres",
    page_icon="📊"
)

# ================================
# CONFIGURACIÓN DE COLORES
# ================================
# Diccionario que asigna un color específico a cada perfil
COLORES_PERFIL = {
    'Perfil 0': "#F5516C",
    'Perfil 1': "#0B2F9A",
    'Perfil 2': "#EA8B6C",
    'Perfil 3': "#72DDF0"
}

# ================================
# DICCIONARIO DE DEPARTAMENTOS
# ================================
# Permite traducir el código del departamento a su nombre real

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

# ================================
# CARGA DEL MAPA GEOJSON
# ================================
# Se utiliza cache para evitar múltiples descargas del archivo
@st.cache_data
def obtener_limites_geojson():
    url = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json"
    try:
        resp = requests.get(url)
        return resp.json()
    except:
        return None

# ================================
# VISUALIZACIÓN DEL PDF DEL ARTÍCULO
# ================================
# Esta función permite mostrar el PDF directamente en la app

def mostrar_pdf_desde_ruta(ruta_pdf):
    if not os.path.exists(ruta_pdf):
        st.error("No se encontró el archivo 'Maternal Profiles Clustering.pdf' en la carpeta del proyecto.")
        return

    with open(ruta_pdf, "rb") as archivo_pdf:
        base64_pdf = base64.b64encode(archivo_pdf.read()).decode("utf-8")

    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="850px" 
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# ================================
# MAPA DE DISPERSIÓN
# ================================
# Genera puntos aleatorios dentro de cada departamento

def generar_mapa_puntos(df, geojson_data):
    if geojson_data is None:
        st.error("No se pudo cargar el archivo GeoJSON del mapa. Verifique su conexión a internet.")
        return px.scatter_mapbox()

    # Se calculan límites geográficos por departamento
    limites = {}

    for feature in geojson_data['features']:
        cod = feature['properties']['DPTO']
        coords = np.array(feature['geometry']['coordinates'][0])

        if coords.ndim == 3:
            coords = coords[0]

        min_lon, min_lat = coords.min(axis=0)
        max_lon, max_lat = coords.max(axis=0)

        m_lon = (max_lon - min_lon) * 0.15
        m_lat = (max_lat - min_lat) * 0.15

        limites[cod] = (
            min_lon + m_lon,
            max_lon - m_lon,
            min_lat + m_lat,
            max_lat - m_lat
        )

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
        df_puntos,
        lat="lat",
        lon="lon",
        color="Perfil",
        color_discrete_map=COLORES_PERFIL,
        hover_name="Departamento",
        zoom=4.7,
        center={"lat": 4.5708, "lon": -74.2973},
        mapbox_style="carto-positron",
        height=700
    )

    fig.update_traces(marker=dict(size=7, opacity=0.7))
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

def generar_mapa_predominancia(df, geojson_data):
    if geojson_data is None:
        st.error("No se pudo cargar el archivo GeoJSON del mapa. Verifique su conexión a internet.")
        return px.choropleth()

    df_local = df.copy()
    df_local['C_Num'] = df_local['Cluster'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

    df_moda = df_local.groupby('CODPTORE')['C_Num'].agg(lambda x: x.mode()[0]).reset_index()
    df_moda['Perfiles'] = df_moda['C_Num'].apply(lambda x: f"Perfil {x}")
    df_moda['Nombre_Dpto'] = df_moda['CODPTORE'].apply(lambda x: NOMBRES_DEPTOS.get(x, "Desconocido"))

    fig = px.choropleth(
        df_moda,
        geojson=geojson_data,
        locations="CODPTORE",
        featureidkey="properties.DPTO",
        color="Perfiles",
        color_discrete_map=COLORES_PERFIL,
        scope="south america",
        hover_name="Nombre_Dpto",
        hover_data={'CODPTORE': True, 'Perfiles': True, 'C_Num': False},
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
    fig.update_traces(marker_line_color="white", marker_line_width=1.0)

    return fig

# ================================
# CONCLUSIONES POR PERFIL
# ================================
# Presenta el análisis interpretativo de los perfiles

def mostrar_conclusiones_por_perfil():
    st.markdown("## Conclusiones por perfil")

    st.markdown("""
### Perfil 0 – Condiciones favorables de atención y resultado neonatal

Este perfil se caracteriza por presentar mejores indicadores en el control prenatal y en las condiciones del nacimiento. Se observa un mayor número de consultas médicas, lo cual sugiere un adecuado seguimiento durante el embarazo. Asimismo, los valores asociados al peso del recién nacido y al tiempo de gestación indican condiciones cercanas a la normalidad.

**Conclusión:**  
Este perfil representa un grupo con bajo nivel de riesgo, asociado a mejores prácticas de cuidado prenatal y posiblemente a un mayor acceso a servicios de salud. Puede considerarse como un referente de condiciones óptimas dentro del análisis.
""")

    st.markdown("---")

    st.markdown("""
### Perfil 1 – Condiciones intermedias con estabilidad relativa

Las madres en este perfil presentan valores promedio en la mayoría de las variables, incluyendo número de consultas prenatales, edad y condiciones del nacimiento. No se evidencian extremos significativos, lo que sugiere un comportamiento relativamente estable.

**Conclusión:**  
Este perfil corresponde a un grupo con riesgo moderado, cuyas condiciones no son críticas pero tampoco óptimas. Representa una población que podría beneficiarse de mejoras en el acceso o calidad del control prenatal.
""")

    st.markdown("---")

    st.markdown("""
### Perfil 2 – Mayor vulnerabilidad y riesgo materno-infantil

Este perfil presenta los indicadores menos favorables, destacándose un menor número de consultas prenatales, menores valores en el peso del recién nacido y menor tiempo de gestación. Estos factores están asociados a un mayor riesgo en términos de salud neonatal.

**Conclusión:**  
Este perfil corresponde a un grupo de alta vulnerabilidad, que probablemente enfrenta barreras en el acceso a servicios de salud o condiciones socioeconómicas desfavorables. Es el perfil que requiere mayor atención y priorización en estrategias de intervención.
""")

    st.markdown("---")

    st.markdown("""
### Perfil 3 – Perfil predominante con condiciones heterogéneas

Este perfil agrupa una proporción importante de los registros, lo que indica que representa condiciones comunes dentro de la población analizada. Sus indicadores suelen ubicarse en rangos intermedios, con cierta variabilidad.

**Conclusión:**  
Este perfil refleja el comportamiento promedio de la población, con condiciones mixtas. No presenta el mayor riesgo, pero tampoco los mejores resultados, por lo que su análisis es clave para diseñar estrategias de cobertura general en salud.
""")

    st.markdown("---")

    st.markdown("""
### Conclusión general del análisis

El modelo de segmentación permite identificar claramente diferencias en las condiciones de salud materna y neonatal, evidenciando la existencia de:

- Un grupo con condiciones favorables (**Perfil 0**)
- Un grupo con alta vulnerabilidad (**Perfil 2**)
- Perfiles intermedios que representan la mayoría de la población (**Perfiles 1 y 3**)

Este tipo de análisis es fundamental para orientar políticas públicas, priorizar recursos y mejorar la atención en salud materno-infantil, enfocándose especialmente en los grupos con mayor riesgo.
""")

# ================================
# SECCIÓN "ACERCA DEL PROYECTO"
# ================================
# Explicación completa del proyecto

def mostrar_acerca_de():
    st.title("ℹ️ Acerca de")

    st.markdown("""
## Proyecto: Identificación de perfiles maternos y salud neonatal en Colombia

Este proyecto tiene como propósito identificar perfiles de madres en Colombia a partir de variables sociodemográficas, clínicas y de atención prenatal, utilizando técnicas de **Machine Learning no supervisado**.

---

## Objetivo general

Identificar patrones en los registros de nacimientos para caracterizar grupos de madres con condiciones similares, permitiendo reconocer perfiles con mejores condiciones, perfiles intermedios y perfiles con mayor vulnerabilidad materno-infantil.

---

## Algoritmos validados

Para el desarrollo del proyecto se utilizaron técnicas de aprendizaje no supervisado, especialmente el algoritmo **K-Means Clustering**.

### K-Means Clustering

Este algoritmo permite agrupar registros similares en diferentes perfiles o clusters. Su uso es adecuado porque:

- Permite identificar grupos naturales dentro de los datos.
- Facilita la interpretación de patrones poblacionales.
- Es eficiente para procesar grandes volúmenes de información.
- Permite segmentar madres según condiciones sociales, clínicas y de atención prenatal.

El modelo permite clasificar los registros en cuatro perfiles principales:

- **Perfil 0:** Condiciones favorables.
- **Perfil 1:** Condiciones intermedias.
- **Perfil 2:** Mayor vulnerabilidad.
- **Perfil 3:** Condiciones heterogéneas o predominantes.

---

## Estructura técnica del proyecto

La aplicación fue desarrollada en **Python** utilizando **Streamlit** como framework principal de visualización.

### Componentes principales

- **Python:** lenguaje base del proyecto.
- **Streamlit:** construcción de la interfaz web interactiva.
- **Pandas:** procesamiento, limpieza y análisis de datos.
- **NumPy:** operaciones numéricas.
- **Plotly:** generación de gráficos y mapas interactivos.
- **Requests:** conexión con recursos externos como archivos GeoJSON.
- **Modelo de Machine Learning:** función `clasificar_madre()` integrada desde el archivo `model.py`.
- **GeoJSON:** visualización geográfica de perfiles por departamento.

---

## Flujo general del sistema

1. El usuario carga un archivo CSV con datos de nacimientos.
2. El sistema realiza limpieza y filtrado de registros.
3. Se eliminan valores inconsistentes o codificados como sin información.
4. El modelo clasifica los registros en perfiles maternos.
5. Se genera una tabla de caracterización por perfil.
6. Se visualizan mapas de dispersión y predominancia regional.
7. Se presentan conclusiones por perfil.
8. Se integra el artículo metodológico del proyecto.

---

## Variables analizadas

El modelo considera variables relacionadas con condiciones maternas, neonatales y sociales:

### Variables maternas

- Edad de la madre.
- Nivel educativo de la madre.
- Área de residencia.
- Régimen de seguridad social.
- Pertenencia étnica.

### Variables del recién nacido

- Peso al nacer.
- Sexo del nacido.
- Tiempo de gestación.
- APGAR.

### Variables del proceso de atención

- Número de consultas prenatales.
- Tipo de parto.

### Variables familiares

- Edad del padre.
- Nivel educativo del padre.

---

## Análisis profesional de los perfiles

### Perfil 0 – Condiciones favorables de atención y resultado neonatal

Este perfil representa madres con mejores condiciones en el control prenatal y en los indicadores asociados al nacimiento. Se caracteriza por un mayor número de consultas médicas, mejores condiciones de peso al nacer y tiempos de gestación cercanos a rangos esperados.

**Interpretación:**  
Es un perfil de bajo riesgo relativo y puede considerarse como referencia de condiciones favorables dentro del análisis.

---

### Perfil 1 – Condiciones intermedias con estabilidad relativa

Este perfil agrupa madres con valores promedio en varias variables. No representa el grupo más vulnerable, pero tampoco evidencia las mejores condiciones.

**Interpretación:**  
Corresponde a un grupo con riesgo moderado, que puede beneficiarse de estrategias de fortalecimiento en acceso y seguimiento prenatal.

---

### Perfil 2 – Mayor vulnerabilidad materno-infantil

Este perfil presenta indicadores menos favorables, especialmente en variables como número de consultas prenatales, peso al nacer y tiempo de gestación.

**Interpretación:**  
Es el perfil que requiere mayor atención, ya que puede estar asociado con barreras de acceso a servicios de salud, condiciones socioeconómicas desfavorables o menor seguimiento durante el embarazo.

---

### Perfil 3 – Condiciones heterogéneas o predominantes

Este perfil puede representar una proporción importante de la población analizada y muestra condiciones mixtas o intermedias.

**Interpretación:**  
Es clave para comprender el comportamiento general de la población y orientar estrategias de cobertura amplia.

---

## Valor investigativo del proyecto

Este proyecto aporta valor porque permite pasar de una visión general de los nacimientos a una segmentación más precisa por perfiles. Esto puede apoyar:

- Identificación de grupos vulnerables.
- Diseño de intervenciones focalizadas.
- Planeación de políticas públicas.
- Priorización de recursos en salud materno-infantil.
- Análisis territorial de condiciones de nacimiento.
- Seguimiento de desigualdades sociales y sanitarias.

---

## Valor para la salud pública

La identificación de perfiles maternos permite orientar decisiones basadas en datos. En lugar de tratar a toda la población de madres como un grupo homogéneo, el sistema permite reconocer diferencias relevantes entre grupos.

Esto facilita:

- Mejor focalización de programas de control prenatal.
- Detección de zonas con mayor vulnerabilidad.
- Fortalecimiento de la atención primaria.
- Priorización de madres con mayor riesgo.
- Diseño de estrategias preventivas.

---

## Conclusión general

El proyecto demuestra que el uso de Machine Learning puede aportar herramientas valiosas para el análisis de la salud materna y neonatal. La segmentación por perfiles permite identificar patrones ocultos en los datos y convertirlos en información útil para la toma de decisiones.

En conclusión, esta aplicación contribuye a una visión más precisa, preventiva y focalizada de la atención materno-infantil en Colombia.
""")

# ================================
# INTERFAZ PRINCIPAL
# ================================
st.title("Identificación de perfiles y condiciones de vida de las madres y la salud neonatal en Colombia")

ruta_imagen = "imagen.png"

if os.path.exists(ruta_imagen):
    st.image(ruta_imagen, use_container_width=True)

st.markdown("---")

with st.sidebar:
    st.header("⚙️ Configuración")
    archivo = st.file_uploader("Cargar dataset de nacimientos (CSV)", type=["csv"])

tab_resumen, tab_puntos, tab_regional, tab_articulo, tab_acerca = st.tabs([
    "📊 Resumen Perfiles",
    "📍 Mapa Dispersión",
    "🗺️ Mapa Regional",
    "📄 Artículo",
    "ℹ️ Acerca de"
])

if archivo is not None:
    df_nuevo = pd.read_csv(archivo, sep=';')
    
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

    with st.spinner('Analizando datos...'):
        resultados = clasificar_madre(df_nuevo)
    
    resultados['CODPTORE'] = pd.to_numeric(
        resultados['CODPTORE'],
        errors='coerce'
    ).fillna(0).astype(int).astype(str).str.zfill(2)

    geojson = obtener_limites_geojson()

    with tab_resumen:
        c1, c2, c3 = st.columns(3)

        c1.metric("Procesados", f"{len(resultados):,}")
        c2.metric("Perfiles", resultados['Cluster'].nunique())
        c3.metric("Dpto Principal", NOMBRES_DEPTOS.get(resultados['CODPTORE'].mode()[0]))

        st.subheader("Caracterización por Perfil")

        vars_cat = ['SEXO', 'TIPO_PARTO', 'AREA_RES', 'SEG_SOCIAL', 'IDPERTET', 'NIV_EDUM']
        vars_num = ['EDAD_MADRE', 'EDAD_PADRE', 'NUMCONSUL', 'T_GES', 'PESO_NAC']
        
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

        res_num = resultados.groupby('Cluster')[vars_num].mean()
        res_cat = resultados.groupby('Cluster')[vars_cat].agg(lambda x: x.mode()[0])

        df_pro = pd.concat([res_num, res_cat], axis=1).reset_index()
        df_pro['Cluster'] = df_pro['Cluster'].astype(str).replace('Cluster ', 'Perfil ', regex=True)
        df_pro.rename(columns={'Cluster': 'Perfil'}, inplace=True)
        
        st.dataframe(
            df_pro.style.format(precision=1).background_gradient(cmap='Blues', subset=vars_num),
            use_container_width=True
        )

        mostrar_conclusiones_por_perfil()

        dist = resultados['Cluster'].value_counts().reset_index()
        dist.columns = ['Perfil', 'Cantidad']
        dist['Perfil'] = dist['Perfil'].astype(str).replace('Cluster ', 'Perfil ', regex=True)

        st.plotly_chart(
            px.bar(
                dist,
                x='Perfil',
                y='Cantidad',
                color='Perfil',
                color_discrete_map=COLORES_PERFIL
            ),
            use_container_width=True
        )

    with tab_puntos:
        st.markdown(
            "Esta visualización cartográfica emplea un algoritmo de distribución estocástica "
            "para representar la densidad poblacional de los perfiles identificados a nivel departamental, "
            "facilitando el análisis de micro-focos de vulnerabilidad o características específicas "
            "de los diferentes perfiles maternos."
        )

        st.plotly_chart(
            generar_mapa_puntos(resultados, geojson),
            use_container_width=True
        )

    with tab_regional:
        st.markdown(
            "Este mapa visualiza la variable de tendencia central (moda estadística) "
            "de los perfiles maternos por departamento"
        )

        st.plotly_chart(
            generar_mapa_predominancia(resultados, geojson),
            use_container_width=True
        )

else:
    with tab_resumen:
        st.info("Cargue un archivo CSV para comenzar.")

    with tab_puntos:
        st.info("Cargue un archivo CSV para visualizar el mapa de dispersión.")

    with tab_regional:
        st.info("Cargue un archivo CSV para visualizar el mapa regional.")

with tab_articulo:
    st.title("📄 Artículo: Maternal Profiles Clustering")

    st.markdown("""
Este apartado presenta el documento **Maternal Profiles Clustering**, integrado directamente en el proyecto.

El artículo describe el enfoque metodológico basado en **Machine Learning no supervisado**, específicamente **K-Means Clustering**, para identificar perfiles maternos y analizar condiciones asociadas a la salud neonatal en Colombia.
""")

    st.markdown("---")

    ruta_articulo = "Maternal Profiles Clustering.pdf"
    mostrar_pdf_desde_ruta(ruta_articulo)

with tab_acerca:
    mostrar_acerca_de()
