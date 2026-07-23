# Replicación COM-B (Valeria)

Réplica de la metodología de Karen (`originales/`) para identificar los **componentes del marco COM-B** (Capacidad, Oportunidad, Motivación — Michie, van Stralen & West, 2011) en el corpus de tweets, siguiendo la misma estructura de las replicaciones `mios/` (salud) y `mios - comportamiento/` (comportamientos protectores).

- **Fuente de datos:** `C:\Users\afpue\Documents\GitHub\icare\archivos\df_twitter.csv` (la misma de las replicaciones anteriores).
- **Salidas:** todas van a `C:\Users\afpue\Documents\GitHub\icare\kMetodo\resultadosCOMB`.
- **Tesauro:** se genera en `resultadosCOMB/Tesauro_COMB/` (NB03) a partir del documento *"Listado de palabras clave del COM-B"* de la magister en estudios del comportamiento. Son **6 subcomponentes**, cada uno con sus palabras clave + ejemplos en contexto de pandemia: `COMB_Motivacion_reflexiva`, `COMB_Motivacion_automatica`, `COMB_Oportunidad_fisica`, `COMB_Oportunidad_social`, `COMB_Capacidad_fisica`, `COMB_Capacidad_psicologica`.

## Orden de ejecución

1. `01_preprocesamiento.ipynb` — limpieza, fechas, chunks (1 tweet = 1 chunk).
2. `02_estadisticos_corpus.ipynb` — estadísticos del corpus completo.
3. `03_embeddings_similitudes.ipynb` — crea el tesauro COM-B, calcula embeddings y similitudes coseno tweet × subcomponente. *Si existen `tweet_embeddings.npy` en `resultadosComportamiento` o `resultadosPropios`, se copian automáticamente (mismo corpus y modelo) y no se recalculan.*
4. `04_seleccion_umbral.ipynb` — **tiene pausa manual**: la Fase A exporta `seleccion_umbral.xlsx` (~60 tweets, 10 por subcomponente) para anotar la columna `valido_comb` (1 = el tweet sí expresa el componente asignado); guardar como `seleccion_umbral_anotado.xlsx` y correr la Fase B, que optimiza τ por argmax(accuracy) y etiqueta el corpus completo.
5. `05_descriptivos_comb.ipynb` — descriptivos del subcorpus COM-B.
6. `06_NER.ipynb` — entidades nombradas + co-ocurrencias entre componentes (por scores, multi-etiqueta). *Reutiliza `general_ner.parquet` de replicaciones anteriores si existe.*
7. `07_bertopic_comb.ipynb` — tópicos sobre el subcorpus COM-B.
8. `08_extraer_frecuencias_POS.ipynb` — lemas de verbos/adjetivos/sustantivos (Stanza).
9. `09_descriptivos_subcategorias_comb.ipynb` — distribución y evolución de los 6 subcomponentes + agregación a los 3 componentes mayores (C, O, M).
10. `10_busqueda_lugares_instituciones.ipynb` — búsqueda literal de departamentos, instituciones, figuras, comportamientos y barreras en el subcorpus COM-B + **marcadores lingüísticos del COM-B** como validación léxica.
11. `11_descriptivos_avanzados_comb.ipynb` — integración NER + POS por subcomponente.

## Diferencias clave frente a `mios - comportamiento`

- El tesauro no describe *temas* sino *procesos del comportamiento*: la anotación del NB04 valida si el tweet **expresa** el componente (una intención → Motivación reflexiva; una barrera de acceso → Oportunidad física), no si menciona palabras del tesauro.
- Las co-ocurrencias (NB06) se calculan con los scores de similitud (el COM-B es multi-etiqueta por naturaleza), no con búsqueda literal de términos.
- NB09 agrega los 6 subcomponentes a los 3 componentes mayores (Capacidad, Oportunidad, Motivación).
