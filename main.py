import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Carga del dataset
data = pd.read_csv("data.csv")

# Exploración inicial
print("\nPrimeras filas del dataset:")
print(data.head())

print("\nInformación general del dataset:")
print(data.info())

print("\nEstadísticas generales del dataset:")
print(data.describe())

# Copia de seguridad del dataset original
df = data.copy()

#Conversión de fechas
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Definir las fechas para los conjuntos de entrenamiento, validación y test
fecha_entrenamiento_validacion = datetime(2011, 11, 8)
fecha_test = datetime(2011, 11, 9)

# Dividir en conjunto de entrenamiento + validación y test
df_train_val = df[df['InvoiceDate'] <= fecha_entrenamiento_validacion]
df_test = df[df['InvoiceDate'] >= fecha_test]

# Crear un conjunto de validación del conjunto de entrenamiento
# Usamos el 80% para entrenamiento y el 20% para validación, con series temporales
train_size = int(len(df_train_val) * 0.8)
df_train = df_train_val[:train_size]
df_val = df_train_val[train_size:]

# Mostrar tamaños de los conjuntos
print(f"\nTamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño del conjunto de validación: {len(df_val)}")
print(f"Tamaño del conjunto de test: {len(df_test)}\n")

# Eliminación de valores erróneos
# Eliminamos productos con precios negativos
df = df[df["UnitPrice"] >= 0]

# Ver los 10 productos con el precio unitario más alto
print(f'\n{df.sort_values(by="UnitPrice", ascending=False).head(10)}\n') 

# Lista de descripciones a eliminar
descripciones_a_eliminar = ["AMAZON FEE", "Manual", "Adjust bad debt", "POSTAGE", "DOTCOM POSTAGE", "CRUK Commission", "Bank Charges", "SAMPLES"]

# Filtrar el DataFrame
print(f"Filas antes: {len(df)}")
descripciones_set = set(descripciones_a_eliminar)
df = df[~df["Description"].isin(descripciones_set)]
print(f"Filas después: {len(df)}")

print(f'\n{df.sort_values(by="UnitPrice", ascending=False).head(10)}\n') 

#Eliminación de duplicados
print(f"\nRegistros duplicados antes de eliminar: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Registros duplicados después de eliminar: {df.duplicated().sum()}")

#Manejo de valores nulos
print("\nValores nulos por columna antes de limpieza:")
print(df.isnull().sum())

# Eliminamos artículos sin descripción y sin precio
df = df.dropna(subset=["Description", "UnitPrice"])

# Asignamos un identificador 0 a ventas sin CustomerID
df["CustomerID"] = df["CustomerID"].fillna('NR').astype(str)

print("\nValores nulos después de limpieza:")
print(df.isnull().sum())

#Transformación de datos
# Creación de TotalPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Filtrar los descuentos (StockCode == "D")
descuentos = df[df["StockCode"] == "D"]

# Filtrar devoluciones: InvoiceNo empieza con "C", Quantity < 0 y NO es un descuento
devoluciones = df[(df["InvoiceNo"].str.startswith("C", na="False")) & (df["Quantity"] < 0) & (~df.index.isin(descuentos.index))]

# Calcular total de ventas (sin devoluciones)
ventas = df[~df.index.isin(devoluciones.index)].copy()
ventas["Total"] = ventas["Quantity"] * ventas["UnitPrice"]
total_ventas = ventas["Total"].sum()

# Calcular total de devoluciones
devoluciones = devoluciones.copy() 
devoluciones["Total"] = devoluciones["Quantity"] * devoluciones["UnitPrice"]
total_devoluciones = devoluciones["Total"].sum()

# Calcular ventas netas
ventas_netas = total_ventas - total_devoluciones

# Crear un DataFrame sin devoluciones pero manteniendo los descuentos
df = df[~df.index.isin(devoluciones.index)].copy()

print(f"\nTotal de ventas antes de devoluciones: {total_ventas}")
print(f"Total de devoluciones: {total_devoluciones}")
print(f"Total de ventas netas: {ventas_netas}")

#DataFrame limpio y listo para análisis
print("\nDataFrame final después de limpieza:")
print(df.info())

#Total Vendido por País
plt.figure(figsize=(10, 6))
df_total_por_pais = df.groupby('Country')['TotalPrice'].sum().reset_index()

sns.barplot(x='TotalPrice', y='Country', data=df_total_por_pais.sort_values('TotalPrice', ascending=False).head(10))
plt.title('Total Vendido por País')
plt.xlabel('Total Vendido')
plt.ylabel('País')
plt.tight_layout()

#Agrupar por 'StockCode' (nombre del producto) y sumar la cantidad vendida
productos_mas_vendidos = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
productos_mas_vendidos.plot(kind='bar', color='skyblue')
plt.title('Top 10 Artículos Más Vendidos', fontsize=16)
plt.xlabel('Artículo', fontsize=12)
plt.ylabel('Cantidad Vendida', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()