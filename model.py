import os
import pandas as pd
from pycaret.clustering import load_model, predict_model

# Función para manejar rutas
def get_resource_path(relative_path):
    base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# CARGA DEL MODELO
path = get_resource_path('kmean_pipeline')
model = load_model(path)

# FUNCIÓN DE INFERENCIA
def clasificar_madre(datos_entrada):
    # 'datos_entrada' será el DataFrame que enviaremos desde Streamlit
    datos_entrada = datos_entrada.astype('float64')
    predicciones = predict_model(model, data=datos_entrada)
    
    # Retornamos el resultado para mostrarlo en la App
    return predicciones
