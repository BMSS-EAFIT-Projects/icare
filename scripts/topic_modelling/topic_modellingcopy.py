import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import spacy
import torch

# === CONFIGURACIÓN ===
ARCHIVO_ENTRADA = "/home/afpuertav/mySpace/archivos/ds68_original.xlsx"
CARPETA_SALIDA = "/home/afpuertav/mySpace/archivos/topicos"
ARCHIVO_SALIDA = f"{CARPETA_SALIDA}/topics_68_lemmatized.csv"
COLUMNA_TEXTO = "Content"
NUM_TOPICOS = 20
MIN_TOPIC_SIZE = 1000

# === Descargar recursos necesarios ===
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("spanish"))

# === Cargar modelo spaCy para lematización en español ===
nlp = spacy.load("es_core_news_md")

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\\S+|www\\S+|https\\S+", "", texto)
    texto = re.sub(r"@\\w+", "", texto)
    texto = re.sub(r"#", "", texto)
    texto = re.sub(r"[^a-záéíóúüñ\\s]", "", texto)
    texto = re.sub(r"\\s+", " ", texto).strip()
    return texto

def lematizar(texto):
    doc = nlp(texto)
    return " ".join([token.lemma_ for token in doc if token.lemma_ not in stop_words and len(token.lemma_) > 2])

# === Cargar datos ===
print("Cargando datos...")
df = pd.read_excel(ARCHIVO_ENTRADA)
df = df[df[COLUMNA_TEXTO].notna()].copy()

# === Preprocesar texto ===
print("Lematizando y limpiando textos...")
df["Texto_Limpio"] = df[COLUMNA_TEXTO].astype(str).apply(limpiar_texto).apply(lematizar)
docs = df["Texto_Limpio"].tolist()

# === Modelo de embeddings ===
print("Cargando modelo de embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embedding_model.to(device)

# === Crear modelo BERTopic ===
print("Entrenando modelo BERTopic...")
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=list(stop_words))

topic_model = BERTopic(
    embedding_model=embedding_model,
    language="multilingual",
    calculate_probabilities=True,
    nr_topics=NUM_TOPICOS,
    min_topic_size=MIN_TOPIC_SIZE,
    vectorizer_model=vectorizer_model
)

topics, probs = topic_model.fit_transform(docs)

# === Guardar resultados ===
df["Topic"] = topics
df["Topic_Prob"] = [probs[i][t] if t != -1 else None for i, t in enumerate(topics)]
df.to_csv(ARCHIVO_SALIDA, index=False)
print(f"Resultados guardados en: {ARCHIVO_SALIDA}")

# === Exportar resumen e informes ===
df_summary = topic_model.get_topic_info()
df_summary.to_csv(f"{CARPETA_SALIDA}/resumen_topics.csv", index=False)
topic_model.save(f"{CARPETA_SALIDA}/modelo_bertopic")

print("Proceso completado.")