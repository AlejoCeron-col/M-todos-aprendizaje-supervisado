# Sistema de Clasificación de Rutas de Transporte Masivo

## Descripción

Este proyecto implementa un modelo de aprendizaje automático supervisado para clasificar recorridos de un sistema de transporte masivo en dos categorías: **rutas rápidas y rutas lentas**.

Se utiliza un conjunto de datos (dataset) con información sobre estaciones de origen y destino, líneas de transporte, distancia, tiempo de recorrido, cantidad de pasajeros y retrasos. El modelo permite evaluar su rendimiento y predecir la categoría de nuevos recorridos.

## Tecnologías utilizadas

* **Python:** lenguaje de programación.
* **Pandas y NumPy:** manejo y procesamiento de datos.
* **Matplotlib:** visualización de datos mediante un gráfico de dispersión.
* **Scikit-learn:** entrenamiento, evaluación y predicción con un árbol de decisión.

## Archivos del proyecto

* `Actividad 3.py`: crea el dataset, entrena el modelo, evalúa su precisión, genera un gráfico de dispersión y predice nuevos recorridos.

## Instalación y ejecución

**1. Instalar Python**

Descarga Python desde [python.org](https://www.python.org/downloads/).

**2. Instalar las dependencias**

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

**3. Ejecutar el proyecto**

```bash
python Actividad 3.py
```

El programa mostrará los datos, las métricas de evaluación y las predicciones, además del gráfico de dispersión.


## Funcionamiento del modelo

1. Se crea un dataset de 30 recorridos de ejemplo.
2. Se convierten las variables de texto en valores numéricos.
3. Se separan los datos en entrenamiento (70 %) y prueba (30 %).
4. Se entrena un árbol de decisión para clasificar las rutas.
5. Se evalúa el modelo mediante la precisión y el reporte de clasificación.
6. Se genera un gráfico que compara la distancia y el tiempo de los recorridos.
7. Se clasifican nuevos recorridos como rápidos o lentos.
