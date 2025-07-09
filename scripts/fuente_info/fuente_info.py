import pandas as pd

ruta_archivo = "/home/afpuertav/mySpace/archivos/tweetsCompletos.xlsx"

xls = pd.ExcelFile(ruta_archivo)
df = pd.concat([xls.parse(hoja) for hoja in xls.sheet_names], ignore_index=True)

col_sentimiento = [col for col in df.columns if col.lower() == 'sentimiento']
if col_sentimiento:
    col_sentimiento = col_sentimiento[0]
else:
    raise ValueError("No se encontró una columna llamada 'Sentimiento' o 'sentimiento'")

mapa_sentimiento = {
    '1 star': 'Negative',
    '2 stars': 'Negative',
    '3 stars': 'Neutral',
    '4 stars': 'Positive',
    '5 stars': 'Positive'
}
df[col_sentimiento] = df[col_sentimiento].map(mapa_sentimiento)

output_path = "/home/afpuertav/mySpace/archivos/tweetsCompletos_p1.csv"
df.to_csv(output_path, index=False)

print(f"Archivo guardado en: {output_path}")
