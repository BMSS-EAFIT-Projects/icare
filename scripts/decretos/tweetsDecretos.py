import os
import glob
import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, pipeline

# Verificar si hay GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Usando dispositivo: {'GPU' if device == 0 else 'CPU'}")

carpeta = "/home/afpuertav/mySpace/archivos/lotesSentido"
archivos_excel = sorted(glob.glob(os.path.join(carpeta, '*.xlsx')))
df = pd.concat([pd.read_excel(archivo) for archivo in archivos_excel], ignore_index=True)
df = df[df['Content_limpio'].str.contains('circular', case=False, na=False)].copy()

model_path = "/home/afpuertav/.cache/huggingface/hub/models--cardiffnlp--twitter-xlm-roberta-base-sentiment/snapshots/f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"

# ✅ Importación corregida para usar el tokenizador lento
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

percepciones = []
batch_size = 32

for i in range(0, len(df), batch_size):
    batch_texts = df['Content_limpio'].iloc[i:i + batch_size].tolist()
    results = sentiment_pipeline(batch_texts)
    for res in results:
        label = res['label']
        if label == 'positive':
            percepciones.append('positiva')
        elif label == 'negative':
            percepciones.append('negativa')
        else:
            percepciones.append('neutra')

df['Percepcion_Decreto'] = percepciones

salida = "/home/afpuertav/mySpace/archivos/medidas/hablandoCircular.xlsx"
df.to_excel(salida, index=False)

print(f"✅ Archivo guardado exitosamente en: {salida}")
