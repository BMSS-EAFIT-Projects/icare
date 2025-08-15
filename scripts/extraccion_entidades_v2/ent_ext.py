import pandas as pd
import torch
from transformers import pipeline
from collections import defaultdict
from tqdm import tqdm

# Datos de entrada y salida
ARCHIVO_ENTRADA = "/home/afpuertav/mySpace/archivos/entidades_68.csv"
ARCHIVO_SALIDA = "/home/afpuertav/mySpace/archivos/entidades_68_V2.csv"
COLUMNA_TEXTO = "Author Description"
BATCH_SIZE = 128

# Asignación del dispositivo para PyTorch
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli",
                      device=device)

# Bolsa de etiquetas (sin la OMS para evitar volver a clasificarla)
bolsa_etiquetas = {
    "Noticias locales, nacionales o globales": ["Noticias locales, nacionales o globales","noticias", "periodista", "radio", "TV", "canal",
                                                  "periódico", "noticiero", "prensa", "televisión",
                                                  "reportero", "medio de comunicación"],
    "Doctor o profesional de salud": ["Doctor o profesional de salud","doctor", "médico", "salud", "hospital", "clínica"],
    "Autoridad local de salud o gobierno": ["Autoridad local de salud o gobierno","ministerio", "alcaldía", "gobierno", "INS", "secretaría de salud",
                                            "eps", "minsaldo", "entidad de salud", "INVIMA"],
    "Líder nacional": ["Líder nacional","presidente", "senador", "diputado", "ministro"],
    "Líder cultural, religioso o comunitario": ["Líder cultural, religioso o comunitario","pastor", "sacerdote", "comunidad", "iglesia", "líder cultural",
                                                "líder religioso", "líder comunitario", "organización cultural"],
    "Literatura científica": ["Literatura científica","investigador", "PhD", "ciencia", "artículo", "publicación", "cientifico"],
}

def clasificar_batch(text_batch, bolsa):
    resultados = []
    for texto in text_batch:
        if not isinstance(texto, str) or texto.strip() == "":
            resultados.append(("Sin descripción", 0.0))
            continue

        scores_categoria = defaultdict(float)
        for categoria, palabras in bolsa.items():
            res = classifier(texto, palabras, multi_label=True)
            for label, score in zip(res['labels'], res['scores']):
                scores_categoria[categoria] += score

        suma = sum(scores_categoria.values())
        if suma == 0:
            resultados.append(("Sin categoría", 0.0))
        else:
            categoria_max = max(scores_categoria, key=scores_categoria.get)
            probabilidad = scores_categoria[categoria_max] / suma
            resultados.append((categoria_max, round(probabilidad, 4)))
    return resultados

def aplicar_por_lotes(df, columna_texto, batch_size):
    entidades = []
    probabilidades = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[columna_texto].iloc[i:i+batch_size].tolist()
        resultado_batch = clasificar_batch(batch, bolsa_etiquetas)
        entidades.extend([r[0] for r in resultado_batch])
        probabilidades.extend([r[1] for r in resultado_batch])
    df["Entidad"] = entidades
    df["Prob entidad"] = probabilidades
    return df

def main():
    print("Cargando archivo...")
    df = pd.read_csv(ARCHIVO_ENTRADA)

    # Filtrar solo los mal clasificados
    df_oms = df[df["Entidad"] == "Organización Mundial de la Salud"].copy()
    if df_oms.empty:
        print("No hay registros con 'Organización Mundial de la Salud'.")
        return

    print(f"Reclasificando {len(df_oms)} registros...")
    df_oms = aplicar_por_lotes(df_oms, COLUMNA_TEXTO, BATCH_SIZE)

    # Actualizar el DataFrame original
    df.update(df_oms)

    print("Guardando resultados...")
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"Archivo guardado en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()
