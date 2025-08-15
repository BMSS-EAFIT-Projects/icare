import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

# ========= CONFIGURACIÓN =========
ARCHIVO_ENTRADA = "/home/afpuertav/mySpace/archivos/ds68_original.xlsx"
ARCHIVO_SALIDA = "/home/afpuertav/mySpace/archivos/polaridad_68.csv"
COLUMNA_TEXTO = "Content"

# ========= CARGAR MODELO =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

labels = ['Negativa', 'Neutra', 'Positiva']

# ========= CLASIFICACIÓN =========
def clasificar_polaridad(texto):
    if not isinstance(texto, str) or texto.strip() == "":
        return "Sin contenido"

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits.cpu().numpy(), axis=1)[0]
        pred = labels[probs.argmax()]
    return pred

# ========= PROCESAMIENTO =========
def main():
    print("Leyendo archivo...")
    df = pd.read_excel(ARCHIVO_ENTRADA)

    print("Clasificando polaridad...")
    tqdm.pandas()
    df["Polaridad"] = df[COLUMNA_TEXTO].progress_apply(clasificar_polaridad)

    print("Guardando resultados...")
    df.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"Archivo guardado en: {ARCHIVO_SALIDA}")

if __name__ == "__main__":
    main()
