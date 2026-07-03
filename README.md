# App web: intersecciones y tramos viales críticos (multi-ciudad)

Aplicación en Streamlit que, a partir de un CSV de accidentes y el nombre de una
ciudad, descarga la red vial de OpenStreetMap, geocodifica los siniestros por
nomenclatura colombiana, construye características de red y entorno, entrena un
clasificador con **validación cruzada espacial** y despliega mapas y métricas.

## Archivos
- `app.py` — la aplicación.
- `requirements.txt` — dependencias.

## Formato del CSV
Nueve columnas, en este orden (o con estos nombres): fecha (`"%Y %b %d %I:%M:%S %p"`),
código, dirección, barrio, vehículos, heridos, gravedad (`con heridos` / `con muertos`
/ `solo daños`), muertes (admite `NO APLICA` = 0), clase.

## Ejecutar en local
```bash
pip install -r requirements.txt
streamlit run app.py
```
Abre http://localhost:8501

## Desplegar gratis

### Opción A — Streamlit Community Cloud
1. Sube `app.py` y `requirements.txt` a un repositorio de GitHub.
2. Entra a https://share.streamlit.io , conecta el repo y despliega.
3. Nota: el plan gratuito tiene ~1 GB de RAM; suficiente para ciudades pequeñas
   e intermedias. Si una ciudad grande agota memoria, usa Hugging Face Spaces.

### Opción B — Hugging Face Spaces (más holgado)
1. Crea un Space tipo **Streamlit**.
2. Sube `app.py` y `requirements.txt`.
3. Arranca solo (aguanta mejor las librerías geoespaciales pesadas).

## Advertencias de rendimiento (modo multi-ciudad)
- **Primera ejecución de una ciudad:** descarga la red de OSM y calcula la
  centralidad de intermediación (betweenness), lo más costoso. Puede tardar
  **1–5 min** en ciudades pequeñas y **más** en grandes. Los resultados quedan
  **en caché** (`st.cache_resource`): reejecutar la misma ciudad es casi instantáneo.
- **Overpass (servidor de OSM)** ocasionalmente limita o rechaza peticiones:
  si falla, reintenta pasados unos segundos.
- **Betweenness** es O(n·m): en metrópolis puede tardar varios minutos o agotar
  memoria en planes gratuitos. Para ciudades muy grandes, considera precomputar
  y cachear en disco.
- El **módulo de visión satelital se omite**: era pesado y no mejoró el modelo
  en el estudio; la app usa el modelo tabular (final).
- El **modelo se entrena por ciudad** con los siniestros del CSV subido (deriva la
  etiqueta `crítico = ≥1 siniestro`). Requiere suficientes casos positivos: con
  muy pocos, la validación espacial se vuelve inestable (la app lo advierte).

## Limitaciones metodológicas (heredadas del estudio)
- La betweenness es un *proxy* de exposición (forma de la red, no aforos reales).
- Posible subregistro policial; direcciones rurales por kilometraje se excluyen.
- Los puntajes son de **ordenamiento/priorización**, no probabilidades calibradas
  (por eso los mapas usan percentiles).
- La calidad depende de la cobertura de OSM en la ciudad indicada.
