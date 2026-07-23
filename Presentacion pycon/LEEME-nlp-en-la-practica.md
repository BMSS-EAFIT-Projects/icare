# NLP en la práctica — versión Quarto (nativa y editable)

Reproducción del PowerPoint `NLP en la practica - PyCon 2026.pptx` como presentación
**Quarto / reveal.js con texto real** (no imágenes de diapositivas). Todo el contenido
—títulos, tarjetas, tablas, líneas de tiempo, chips— es HTML/CSS editable, así que puedes
cambiar textos, colores o reordenar sin volver a PowerPoint.

## Archivos

```
quarto/
├── nlp-en-la-practica.qmd     # la presentación (41 diapositivas)
├── theme-nlp.scss             # tema con la paleta y componentes de PyCon 2026
└── img_nlp/                    # imágenes reales extraídas del pptx (fotos, gráficos, QR)
```

La versión anterior basada en imágenes de fondo (`slides_img/`) queda reemplazada por esta;
esa carpeta ya no se usa.

## Requisitos

- **Quarto** — https://quarto.org/docs/get-started/ (en Windows: `winget install quarto`).
- En VS Code, la extensión **Quarto** (botón *Preview/Render*).
- No necesita Python.

## Previsualizar / exportar

```bash
cd quarto
quarto preview nlp-en-la-practica.qmd     # ver en vivo con recarga
quarto render  nlp-en-la-practica.qmd     # genera nlp-en-la-practica.html
```

Para un **único archivo portable** (útil sin internet en el evento), añade bajo `revealjs:`
en el encabezado YAML la línea `embed-resources: true` y vuelve a renderizar.

## Navegación

Flechas ← →, **Esc** vista general, **F** pantalla completa, **S** notas del ponente.

## Fidelidad y notas

- Paleta exacta del pptx: violeta `#614AD3`, violeta profundo `#3F2F9E`, oro `#FFC24B`,
  rosa `#FF5C8A`, tinta `#1A1633`, fondo `#F2F3FD`.
- Tipografía: usa **Aptos** si está instalada (Office 365); si no, cae a Segoe UI / system.
- Las diapositivas oscuras (portada, separadores, ideas fuerza, cierre) ocultan el pie y el
  número de página, igual que en el pptx.
- Para editar un texto, búscalo en `nlp-en-la-practica.qmd`; para ajustar un color o un
  componente, edita `theme-nlp.scss`.
