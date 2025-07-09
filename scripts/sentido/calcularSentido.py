import pandas as pd
import torch
import math
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# === CONFIGURACIÓN ===
INPUT_FILE = "/home/afpuertav/mySpace/archivos/tweetsCompletos20250612.csv"
OUTPUT_DIR = "/home/afpuertav/mySpace/archivos/lotesSentido"
LOTE_TAMANO = 10000
UMBRAL_PERPLEJIDAD = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === CARGA DEL MODELO GPT-2 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-2 no tiene pad_token por defecto
tokenizer.pad_token = tokenizer.eos_token

model.eval()

# === FUNCIÓN PARA CALCULAR PERPLEJIDAD INDIVIDUAL ===
def calcular_perplejidades_batch(textos):
    resultados = []
    for texto in textos:
        if not isinstance(texto, str) or len(texto.strip()) < 5:
            resultados.append(float("inf"))
            continue
        try:
            inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss
            ppl = math.exp(loss.item())
            resultados.append(ppl)
        except Exception:
            resultados.append(float("inf"))
    return resultados

# === PROCESAR Y GUARDAR CADA LOTE ===
def procesar_lote(df_lote, i):
    print(f"Procesando lote {i} ({len(df_lote)} tweets)...")
    textos = df_lote['Content_limpio'].fillna("").tolist()
    ppls = calcular_perplejidades_batch(textos)
    df_lote['perplejidad'] = ppls

    output_path = os.path.join(OUTPUT_DIR, f"lote_sentido_{i:02d}.xlsx")
    df_lote.to_excel(output_path, index=False)
    print(f"Guardado: {output_path}")

# === EJECUCIÓN PRINCIPAL === 
print("Cargando dataset completo...")
df = pd.read_csv(INPUT_FILE)

print(f"Dividiendo en lotes de {LOTE_TAMANO}...")
lotes = [df[i:i + LOTE_TAMANO] for i in range(0, len(df), LOTE_TAMANO)]

for i, lote in enumerate(lotes):
    procesar_lote(lote.copy(), i)
