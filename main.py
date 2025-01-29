import pandas as pd 
import tabulate as tb

data = pd.read_csv('data.csv')

#Convertimos las fechas a DateTime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

null_description = data[data.Description.isnull()].head()

null_values = data.isnull()

null_by_colum = data.isnull().sum()

print('\nHEAD:\n'+tb.tabulate(data.head(), headers='keys', tablefmt='pretty'))
print('\nTAIL:\n'+tb.tabulate(data.tail(), headers='keys', tablefmt='pretty'))
print('\nDESCRIPCIÓN DEL DATASET:\n'+tb.tabulate(data.describe(),headers='keys', tablefmt='pretty'))
print('\nVALORES NULOS:\n'+tb.tabulate(null_values.head(), headers='keys', tablefmt='pretty'))
print(f'\nVALORES NULOS POR COLUMNA:\n{null_by_colum}\n')
print(f'\nDESCRIPCION NULA:\n{null_description}')
print(f'{data.info()}')