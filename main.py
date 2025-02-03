import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Preprocesamiento y modelado de ML
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

# Modelos de series temporales y ML
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Modelos de Deep Learning
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. Carga y Exploración Inicial
# -----------------------------
data = pd.read_csv("data.csv")

print("\n[1] Primeras filas del dataset:")
print(data.head())

print("\n[1] Información general del dataset:")
print(data.info())

print("\n[1] Estadísticas generales del dataset:")
print(data.describe())

# Crear copia para trabajar
df = data.copy()

# -----------------------------
# 2. Limpieza de Datos
# -----------------------------

# 2.1 Conversión de fechas
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
# Extraer la fecha para uso en agregaciones
df['Date'] = df['InvoiceDate'].dt.date

# 2.2 Conversión de variables categóricas y manejo de nulos
df["Country"] = df["Country"].astype(str)
df["CustomerID"] = df["CustomerID"].fillna("Unknown").astype(str)

# 2.3 Eliminación de valores erróneos: eliminar productos con UnitPrice negativo
df = df[df["UnitPrice"] >= 0]

# 2.4 Eliminación de filas indeseadas según descripción
descripciones_a_eliminar = ["AMAZON FEE", "Manual", "Adjust bad debt", "POSTAGE",
                            "DOTCOM POSTAGE", "CRUK Commission", "Bank Charges", "SAMPLES"]
df = df[~df["Description"].isin(descripciones_a_eliminar)]
print(f"\n[2] Filas después de eliminar descripciones indeseadas: {len(df)}")

# 2.5 Eliminación de duplicados
print(f"\n[2] Registros duplicados antes de eliminar: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"[2] Registros duplicados después de eliminar: {df.duplicated().sum()}")

# 2.6 Eliminación de filas con valores nulos en columnas críticas
df = df.dropna(subset=["Description", "UnitPrice"])
print("\n[2] Valores nulos después de limpieza:")
print(df.isnull().sum())

# -----------------------------
# 3. Transformación de Datos
# -----------------------------

# 3.1 Creación de la variable TotalPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# 3.2 Filtrado de devoluciones y descuentos:
descuentos = df[df["StockCode"] == "D"]

# Consideramos como devoluciones aquellas filas cuya InvoiceNo empieza con "C", Quantity < 0 y que no sean descuentos
devoluciones = df[(df["InvoiceNo"].str.startswith("C", na="False")) &
                  (df["Quantity"] < 0) &
                  (~df.index.isin(descuentos.index))]

# 3.3 Cálculo de totales de ventas y devoluciones
ventas = df[~df.index.isin(devoluciones.index)].copy()
ventas["Total"] = ventas["Quantity"] * ventas["UnitPrice"]
total_ventas = ventas["Total"].sum()

devoluciones = devoluciones.copy()
devoluciones["Total"] = devoluciones["Quantity"] * devoluciones["UnitPrice"]
total_devoluciones = devoluciones["Total"].sum()

ventas_netas = total_ventas - total_devoluciones

# Remover las devoluciones (manteniendo descuentos) del dataset final
df = df[~df.index.isin(devoluciones.index)].copy()

print(f"\n[3] Total de ventas antes de devoluciones: {round(total_ventas, 2)}")
print(f"[3] Total de devoluciones: {round(total_devoluciones, 2)}")
print(f"[3] Total de ventas netas: {round(ventas_netas, 2)}")

# 3.4 Creación de variable "Ventas Diarias" y variables temporales
ventas_diarias = df.groupby('Date')['TotalPrice'].sum().reset_index()
ventas_diarias.rename(columns={'TotalPrice': 'VentasDiarias'}, inplace=True)
ventas_diarias['Date'] = pd.to_datetime(ventas_diarias['Date'])
ventas_diarias['DiaSemana'] = ventas_diarias['Date'].dt.dayofweek + 1
ventas_diarias['DiaMes'] = ventas_diarias['Date'].dt.day
ventas_diarias['Mes'] = ventas_diarias['Date'].dt.month
ventas_diarias['SemanaAno'] = ventas_diarias['Date'].dt.isocalendar().week

print("\n[3] Ejemplo de ventas diarias y variables temporales:")
print(ventas_diarias.head())

# 3.5 Normalización de variables numéricas (aplicamos StandardScaler a título de ejemplo)
df_scaled = df.copy()
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(df[['Quantity', 'UnitPrice', 'TotalPrice']])
df_normalized = pd.DataFrame(X_normalized, columns=['Quantity', 'UnitPrice', 'TotalPrice'])
print(f'\n{df_normalized.head()}')

# -----------------------------
# 4. División en Conjuntos de Entrenamiento, Validación y Test
# -----------------------------

# Es importante usar un índice de fecha para modelos de series temporales.
# Por ejemplo, para ARIMA y Prophet, podemos establecer 'InvoiceDate' como DateTimeIndex.
df_scaled = df.set_index('InvoiceDate')

# Definir fechas para la división según el enunciado
fecha_entrenamiento_validacion = datetime(2011, 11, 8)
fecha_test = datetime(2011, 11, 9)

# Dividir en conjuntos: train+val (hasta 8/11/2011) y test (desde 9/11/2011)
df_train_val = df_scaled[df_scaled.index <= fecha_entrenamiento_validacion]
df_test = df_scaled[df_scaled.index >= fecha_test]

# Dividir train+val en entrenamiento (80%) y validación (20%)
train_size = int(len(df_train_val) * 0.8)
df_train = df_train_val.iloc[:train_size]
df_val = df_train_val.iloc[train_size:]

print(f"\n[4] Tamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"[4] Tamaño del conjunto de validación: {len(df_val)}")
print(f"[4] Tamaño del conjunto de test: {len(df_test)}\n")

# -----------------------------
# 5. Ejemplo de Modelado: Regresión Polinómica
# -----------------------------

# Usaremos el DataFrame de ventas diarias para modelar 'VentasDiarias'
df_diario = ventas_diarias.copy()

# Variables predictoras y objetivo
X = df_diario[['DiaSemana', 'DiaMes', 'Mes', 'SemanaAno']]
y = df_diario['VentasDiarias']

# Escalado de variables predictoras
scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(X)

# Generar características polinómicas de grado 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Dividir en entrenamiento y validación (80-20)
X_train, X_val, y_train, y_val = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones y RMSE
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"\n[5] RMSE en entrenamiento (Regresión Polinómica): {rmse_train:.2f}")
print(f"[5] RMSE en validación (Regresión Polinómica): {rmse_val:.2f}\n")

# -----------------------------
# 6. Ejemplo de Modelado de Series Temporales y Otros Modelos
# -----------------------------

# ARIMA:
# Para ARIMA, es importante tener un DateTimeIndex. Ya hemos establecido el índice en df.
# Usamos la serie de ventas diarias como ejemplo:
# Convertir 'ventas_diarias' a una serie temporal con DateTimeIndex y frecuencia diaria
ventas_diarias_ts = ventas_diarias.set_index('Date')['VentasDiarias']
ventas_diarias_ts = ventas_diarias_ts.asfreq('D')

# Rellenar los huecos (NaN) usando interpolación lineal
ventas_diarias_ts = ventas_diarias_ts.interpolate(method='linear')

# Crear DataFrame con lags
df_lags = ventas_diarias_ts.to_frame()
df_lags['lag_1'] = df_lags['VentasDiarias'].shift(1)
df_lags['lag_7'] = df_lags['VentasDiarias'].shift(7)

# Eliminar filas con valores NaN generados por los lags
df_lags = df_lags.dropna()

# Verificar el rango de fechas
print("Desde:", ventas_diarias_ts.index.min(), "hasta:", ventas_diarias_ts.index.max())

# Dividir la serie en entrenamiento y test
train_size_ts = int(len(df_lags) * 0.8)
train_series = df_lags.iloc[:train_size_ts]['VentasDiarias']
test_series = df_lags.iloc[train_size_ts:]['VentasDiarias']

print("Train series desde:", train_series.index.min(), "hasta:", train_series.index.max())
print("Test series desde:", test_series.index.min(), "hasta:", test_series.index.max())

# Ajustar el modelo ARIMA en la serie de entrenamiento
model_arima = ARIMA(train_series, order=(5, 1, 0))
model_arima_fit = model_arima.fit()

# Predecir los pasos futuros correspondientes al tamaño del conjunto de test
forecast_steps = len(test_series)
y_val_pred_arima = model_arima_fit.forecast(steps=forecast_steps)

# Evaluar el RMSE en el conjunto de test (que ahora no debe tener NaN)
rmse_arima = np.sqrt(mean_squared_error(test_series, y_val_pred_arima))
print(f"RMSE ARIMA: {rmse_arima:.2f}")

# Prophet:
df_prophet = df_diario[['Date', 'VentasDiarias']].rename(columns={'Date': 'ds', 'VentasDiarias': 'y'})
df_prophet_train = df_prophet[df_prophet['ds'] <= fecha_entrenamiento_validacion]
df_prophet_val = df_prophet[df_prophet['ds'] > fecha_entrenamiento_validacion]
model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)
model_prophet.fit(df_prophet_train)
future = model_prophet.make_future_dataframe(periods=len(df_prophet_val))
forecast = model_prophet.predict(future)
y_val_pred_prophet = forecast['yhat'][-len(df_prophet_val):].values
rmse_prophet = np.sqrt(mean_squared_error(df_prophet_val['y'], y_val_pred_prophet))
print(f"RMSE Prophet: {rmse_prophet:.2f}")

# Random Forest:
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_val_pred_rf = model_rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))
print(f"RMSE Random Forest: {rmse_rf:.2f}")

# XGBoost:
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, y_train)
y_val_pred_xgb = model_xgb.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_val_pred_xgb))
print(f"RMSE XGBoost: {rmse_xgb:.2f}")

# LSTM:
# Para el LSTM, se deben crear secuencias. Definimos la función antes de usarla.
def create_sequences(data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i+seq_length])
        y_seq.append(data[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 10
y_train_array = y_train.values  # Convertir a array
y_val_array = y_val.values

X_train_lstm, y_train_lstm = create_sequences(y_train_array, seq_length)
X_val_lstm, y_val_lstm = create_sequences(y_val_array, seq_length)

# Remodelar para que tenga forma (n_samples, seq_length, 1)
X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_val_lstm = X_val_lstm.reshape((X_val_lstm.shape[0], X_val_lstm.shape[1], 1))

# Modelo LSTM con Dropout y EarlyStopping para prevenir overfitting
model_lstm = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),  # Agregamos dropout para regularización
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=1, callbacks=[early_stop])

y_val_pred_lstm = model_lstm.predict(X_val_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_val_lstm, y_val_pred_lstm))
print(f"RMSE LSTM: {rmse_lstm:.2f}")

# Comparación de RMSE entre modelos
print("\n--- Comparaci\u00f3n de RMSE ---")
print(f"RMSE ARIMA: {rmse_arima:.2f}")
print(f"RMSE Prophet: {rmse_prophet:.2f}")
print(f"RMSE Random Forest: {rmse_rf:.2f}")
print(f"RMSE XGBoost: {rmse_xgb:.2f}")
print(f"RMSE LSTM: {rmse_lstm:.2f}")

# Gráfico comparativo de RMSE
plt.figure(figsize=(10, 6))
model_names = ['ARIMA', 'Prophet', 'Random Forest', 'XGBoost', 'LSTM']
rmse_values = [rmse_arima, rmse_prophet, rmse_rf, rmse_xgb, rmse_lstm]
plt.bar(model_names, rmse_values, color='skyblue')
plt.xlabel('Modelos')
plt.ylabel('RMSE')
plt.title('Comparación de RMSE entre Modelos')
plt.show()

# -----------------------------
# 7. Visualizaciones de Errores
# ----------------------------

# Histograma de Errores (Usando RMSE de regresión polinómica como ejemplo)
errores_train = y_train - y_train_pred
errores_val = y_val - y_val_pred

plt.figure(figsize=(8, 5))
sns.histplot(errores_train, bins=30, kde=True, color="blue", label="Entrenamiento")
sns.histplot(errores_val, bins=30, kde=True, color="orange", label="Validación", alpha=0.6)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("Error (Ventas Reales - Ventas Predichas)")
plt.ylabel("Frecuencia")
plt.title("Distribución de Errores del Modelo (Regresión Polinómica)")
plt.legend()
plt.show()

# Gráfico de Dispersión (Real vs. Predicho) para validaci\u00f3n
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_val, y=y_val_pred, alpha=0.6)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='dashed')
plt.xlabel("Ventas Reales")
plt.ylabel("Ventas Predichas")
plt.title("Comparación entre Ventas Reales y Predichas (Validación)")
plt.show()

# Evolución del RMSE (simulación de iteraciones)
rmse_train_epochs = [rmse_train * 1.2, rmse_train * 1.1, rmse_train * 1.05, rmse_train]
rmse_val_epochs = [rmse_val * 1.3, rmse_val * 1.15, rmse_val * 1.1, rmse_val]
epochs = list(range(1, len(rmse_train_epochs) + 1))

plt.figure(figsize=(8, 5))
plt.plot(epochs, rmse_train_epochs, marker='o', linestyle='-', color='blue', label="RMSE Entrenamiento")
plt.plot(epochs, rmse_val_epochs, marker='s', linestyle='-', color='orange', label="RMSE Validaci\u00f3n")
plt.xlabel("Iteraciones")
plt.ylabel("RMSE")
plt.title("Evolución del RMSE durante el Entrenamiento (Simulado)")
plt.legend()
plt.show()
