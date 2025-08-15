import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# === CONFIGURACIÓN ===
ARCHIVO_ENTRADA = "/home/afpuertav/mySpace/archivos/ds68_original.xlsx"
ARCHIVO_SALIDA = "/home/afpuertav/mySpace/archivos/sentimientos_68.csv"
COLUMNA_TEXTO = "Content"
BATCH_SIZE = 32

# === MODELO ===
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli",
                      device=device)

emociones = [
    "Confianza", "Miedo", "Anticipación", "Tristeza",
    "Alegría", "Sorpresa", "Ira", "Disgusto"
]

# === CLASIFICACIÓN ===
def clasificar_emocion(text_batch, labels):
    resultados = []

    for texto in text_batch:
        if not isinstance(texto, str) or texto.strip() == "":
            resultados.append(("Sin contenido", 0.0))
            continue

        res = classifier(texto, labels, multi_label=False)
        emocion = res["labels"][0]
        prob = res["scores"][0]
        resultados.append((emocion, round(prob, 4)))

    return resultados

# === PROCESAMIENTO POR LOTES ===
def aplicar_por_lotes(df, columna_texto, batch_size, labels):
    emociones_out = []
    probs_out = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[columna_texto].iloc[i:i+batch_size].tolist()
        resultado_batch = clasificar_emocion(batch, labels)
        emociones_out.extend([r[0] for r in resultado_batch])
        probs_out.extend([r[1] for r in resultado_batch])

    df["Emocion"] = emociones_out
    df["Prob emocion"] = probs_out
    return df

# === MAIN ===
def main():
    print("Leyendo archivo...")
    df = pd.read_excel(ARCHIVO_ENTRADA)

    print("Clasificando emociones...")
    df = aplicar_por_lotes(df, COLUMNA_TEXTO, BATCH_SIZE, emociones)

    print("Guardando resultados...")
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"Archivo guardado en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()
