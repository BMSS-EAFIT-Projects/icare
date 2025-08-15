import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch

# === CONFIGURACIÓN ===
ARCHIVO_ENTRADA = "/home/afpuertav/mySpace/archivos/ds68_original.xlsx"
ARCHIVO_SALIDA = "/home/afpuertav/mySpace/archivos/topics_68.csv"
COLUMNA_TEXTO = "Content"

# === Cargar datos ===
print("Cargando datos...")
df = pd.read_excel(ARCHIVO_ENTRADA)
docs = df[COLUMNA_TEXTO].dropna().astype(str).tolist()

# === Modelo de embeddings multilingüe ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cargando modelo de embeddings...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedding_model.to(device)

# === Crear y entrenar el modelo de tópicos ===
print("Entrenando BERTopic...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="multilingual",
    calculate_probabilities=True,
    min_topic_size=1000,      # mínimo de 1000 documentos por tópico
    top_n_words=10
)

topics, probs = topic_model.fit_transform(docs)

# === Guardar resultados en DataFrame ===
df = df.loc[df[COLUMNA_TEXTO].notna()].copy()
df["Topic"] = topics
df["Topic_Prob"] = probs

# === Guardar archivo ===
print("Guardando resultados...")
df.to_csv(ARCHIVO_SALIDA, index=False)
print(f"Resultados guardados en: {ARCHIVO_SALIDA}")

# === Mostrar resumen de tópicos ===
print("\nResumen de tópicos generados:")
print(topic_model.get_topic_info().head(10))  # muestra los primeros 10 tópicos

# === (Opcional) Visualización interactiva ===
# topic_model.visualize_topics().write_html("topics_visual.html")
