import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# Configuración
input_file = '/home/afpuertav/mySpace/archivos/ds68.xlsx'
output_file = '/home/afpuertav/mySpace/archivos/fuente_infoso/result_bert_spanish.xlsx'
batch_size = 16

# Cargar datos
df = pd.read_excel(input_file)

# Eliminar filas con Content_limpio vacío o no string
df = df[df['Content_limpio'].notna()]
df = df[df['Content_limpio'].apply(lambda x: isinstance(x, str))]

# Detectar GPU
device = 0 if torch.cuda.is_available() else -1
print("✅ Usando GPU" if device == 0 else "⚠️ Usando CPU")

# Inicializar modelo
classifier = pipeline("zero-shot-classification", model="Recognai/bert-base-spanish-wwm-cased-xnli", device=device)

# Etiquetas
labels = [
    "literatura científica",
    "organización mundial de la salud",
    "noticias locales, nacionales e internacionales",
    "redes sociales o grupos en internet",
    "doctores y otros profesionales de la salud",
    "otros medios de información",
    "familia, amigos o colegas",
    "lugar de trabajo",
    "gobierno y autoridades de salud pública",
    "líder religioso, cultural o de la comunidad",
    "líder nacional"
]

# Resultados
all_sources = []
all_probs = []

# Procesar por batches
for start in tqdm(range(0, len(df), batch_size)):
    end = min(start + batch_size, len(df))
    batch_texts = df['Content_limpio'].iloc[start:end].tolist()
    batch_texts = [t for t in batch_texts if isinstance(t, str)]
    
    results = classifier(batch_texts, candidate_labels=labels, hypothesis_template="Este texto proviene de {}.")
    
    if isinstance(results, dict):  # caso single
        results = [results]
    
    for res in results:
        probs = {lbl: score for lbl, score in zip(res['labels'], res['scores'])}
        best = max(probs, key=probs.get)
        all_sources.append(best)
        all_probs.append(str(probs))  # guardar como string para Excel

# Agregar columnas al DataFrame original
df['infoso-bert-spanish'] = all_sources
df['prob-infoso-bert-spanish'] = all_probs

# Guardar resultado en Excel
df.to_excel(output_file, index=False)
print(f"✅ Guardado exitosamente en: {output_file}")
