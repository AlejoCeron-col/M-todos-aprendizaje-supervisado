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
