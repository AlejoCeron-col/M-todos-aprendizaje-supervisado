# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS MARCO
# ============================================================
# Pandas: permite crear y manipular tablas de datos.
import pandas as pd
# NumPy: permite trabajar con datos numéricos.
import numpy as np
# Matplotlib: permite crear gráficos.
import matplotlib.pyplot as plt
# train_test_split:
# Divide nuestros datos en datos para entrenar y datos para probar.
from sklearn.model_selection import train_test_split
# LabelEncoder:
# Convierte datos de texto en números.
# Ejemplo:
# "Terminal" -> 3
# "Centro"   -> 1
from sklearn.preprocessing import LabelEncoder
# DecisionTreeClassifier:
# Es el algoritmo de aprendizaje automático que utilizaremos.
from sklearn.tree import DecisionTreeClassifier
# Funciones para evaluar nuestro modelo.
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 2. CREACIÓN DEL DATASET MARCO
# ============================================================

# Creamos un diccionario donde cada columna representa
# una característica de un recorrido.

datos ={

    "origen":[
    
        "Terminal", "Terminal", "Centro", "Centro", "Parque",
        "Parque", "Universidad", "Estadio", "Estadio", "Hospital",
        "Terminal", "Centro", "Parque", "Universidad", "Hospital",
        "Terminal", "Centro", "Estadio", "Parque", "Aeropuerto",
        "Terminal", "Centro", "Universidad", "Estadio", "Hospital",
        "Terminal", "Parque", "Universidad", "Estadio", "Aeropuerto"
],

    # Estación donde termina el recorrido.
    "destino": [
        "Centro", "Estadio", "Parque", "Universidad", "Universidad",
        "Hospital", "Estadio", "Hospital", "Aeropuerto", "Aeropuerto",
        "Parque", "Hospital", "Estadio", "Hospital", "Terminal",
        "Universidad", "Aeropuerto", "Terminal", "Aeropuerto", "Centro",
        "Hospital", "Aeropuerto", "Centro", "Aeropuerto", "Centro",
        "Aeropuerto", "Hospital", "Terminal", "Hospital", "Universidad"
    ],

    # Línea de transporte utilizada.
    "linea": [
        "L1", "L2", "L1", "L1", "L1",
        "L3", "L2", "L2", "L4", "L4",
        "L1", "L3", "L2", "L3", "L3",
        "L1", "L4", "L4", "L2", "L4",
        "L3", "L4", "L1", "L2", "L3",
        "L4", "L2", "L3", "L2", "L4"
    ],

    # Número de estaciones que tiene el recorrido.
    "num_estaciones": [
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
        3, 3, 3, 3, 3,
        4, 4, 4, 4, 4,
        4, 4, 4, 4, 4
    ],

    # Distancia aproximada del recorrido en kilómetros.
    "distancia_km": [
        2.1, 3.5, 1.8, 3.2, 2.7,
        3.8, 3.1, 4.0, 5.2, 5.8,
        4.5, 5.1, 4.3, 5.5, 5.7,
        5.0, 6.2, 6.8, 5.9, 7.0,
        7.2, 8.0, 7.5, 8.3, 8.7,
        9.1, 9.5, 10.0, 9.8, 10.5
    ],

    # Hora en la que comienza el recorrido.
    "hora_salida": [
        7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 21,
        7, 8, 9, 10, 11,
        12, 13, 14, 15, 16,
        17, 18, 19, 20, 21
    ],

    # Tiempo que tarda el recorrido en minutos.
    "tiempo_min": [
        8, 12, 7, 11, 10,
        13, 11, 14, 18, 20,
        15, 17, 16, 19, 21,
        17, 22, 23, 21, 25,
        26, 29, 28, 31, 33,
        32, 35, 37, 36, 40
    ],

    # Cantidad estimada de pasajeros.
    "cantidad_pasajeros": [
        15, 22, 18, 20, 25,
        28, 30, 35, 40, 42,
        45, 50, 48, 55, 60,
        52, 58, 65, 62, 70,
        75, 80, 78, 85, 90,
        88, 95, 92, 100, 105
    ],

    # Cantidad de minutos de retraso.
    "retraso_min": [
        0, 1, 0, 1, 1,
        2, 1, 2, 3, 3,
        2, 3, 2, 4, 4,
        3, 4, 5, 4, 5,
        5, 6, 5, 7, 7,
        6, 8, 7, 9, 10
    ],

    # VARIABLE OBJETIVO.
    #
    # Esta es la respuesta que queremos que la IA aprenda.
    #
    # 0 = Rápida
    # 1 = Lenta
    #
    "tipo_ruta": [
        "Rapida", "Rapida", "Rapida", "Rapida", "Rapida",
        "Rapida", "Rapida", "Rapida", "Lenta", "Lenta",
        "Lenta", "Lenta", "Lenta", "Lenta", "Lenta",
        "Lenta", "Lenta", "Lenta", "Lenta", "Lenta",
        "Lenta", "Lenta", "Lenta", "Lenta", "Lenta",
        "Lenta", "Lenta", "Lenta", "Lenta", "Lenta"
    ]
}

# ============================================================
# 3. CREAR DATAFRAME MARCO
# ============================================================

# Convertimos el diccionario anterior en una tabla de Pandas.
df = pd.DataFrame(datos)

# Mostramos las primeras filas.
print("\n========== DATASET ==========")
print(df.head())

# Mostramos cuántos registros tenemos.
print("\nCantidad de registros:", len(df))

# ============================================================
# 4. CONVERTIR DATOS DE TEXTO A NÚMEROS MARCO
# ============================================================

# Creamos un codificador para cada columna de texto.

le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_linea = LabelEncoder()
le_tipo = LabelEncoder()


# Convertimos el origen.
df["origen"] = le_origen.fit_transform(df["origen"])

# Convertimos el destino.
df["destino"] = le_destino.fit_transform(df["destino"])

# Convertimos la línea.
df["linea"] = le_linea.fit_transform(df["linea"])

# Convertimos nuestra variable objetivo.
df["tipo_ruta"] = le_tipo.fit_transform(df["tipo_ruta"])


# Mostramos el resultado.
print("\n========== DATOS CODIFICADOS ==========")
print(df.head())
# ============================================================
# 5. DEFINIR VARIABLES DE ENTRADA Y VARIABLE OBJETIVO DANIEL
# ============================================================

# X contiene las características que utilizaremos
# para que el modelo aprenda.

X = df[
    [
        "origen",
        "destino",
        "linea",
        "num_estaciones",
        "distancia_km",
        "hora_salida",
        "tiempo_min",
        "cantidad_pasajeros",
        "retraso_min"
    ]
]


# y contiene la respuesta que queremos predecir.

y = df["tipo_ruta"]


print("\n========== VARIABLES ==========")

print("\nVariables de entrada X:")
print(X.head())

print("\nVariable objetivo y:")
print(y.head())

# ============================================================
# 6. DIVIDIR DATOS PARA ENTRENAMIENTO Y PRUEBA DANIEL
# ============================================================

# 70% de los datos serán utilizados para entrenar.
# 30% serán utilizados para comprobar si el modelo aprendió.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\n========== DIVISIÓN DE DATOS ==========")

print("Datos para entrenamiento:", len(X_train))
print("Datos para prueba:", len(X_test))

# ============================================================
# 7. CREAR EL MODELO DE APRENDIZAJE AUTOMÁTICO DANIEL
# ============================================================

# Creamos un árbol de decisión.
#
# max_depth=4 limita la profundidad del árbol.
# Esto evita que el árbol sea excesivamente complejo.

modelo = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


# Entrenamos el modelo utilizando los datos de entrenamiento.

modelo.fit(X_train, y_train)


print("\n========== MODELO ENTRENADO ==========")
print("El árbol de decisión ha aprendido de los datos.")

# ============================================================
# 8. REALIZAR PREDICCIONES ALEJO
# ============================================================

# Utilizamos los datos de prueba.
#
# El modelo NO utilizó estos datos para aprender.

y_pred = modelo.predict(X_test)


print("\n========== PREDICCIONES ==========")

print("Valores reales:")
print(y_test.values)

print("\nValores predichos:")
print(y_pred)
