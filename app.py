# ============================================================
# App web multi-ciudad: identificación de intersecciones y tramos críticos.
# Sube un CSV de accidentes (mismas columnas que Acacías) e indica la ciudad;
# la app descarga la red vial de OSM, geocodifica, construye características,
# entrena con validación cruzada espacial y despliega mapas y métricas.
#
# Ejecutar en local:  streamlit run app.py
# Desplegar:          Streamlit Community Cloud o Hugging Face Spaces (ver README).
# ============================================================
import re, unicodedata, math, io, os, pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Intersecciones críticas", layout="wide")

# Caché en disco para sobrevivir reinicios/suspensión del plan gratuito
CACHE_DIR = ".cache_ciudades"
os.makedirs(CACHE_DIR, exist_ok=True)

# Ciudades frecuentes para el desplegable (elige «Otra…» para escribir cualquiera de OSM)
CIUDADES = [
    "Acacías, Meta, Colombia", "Villavicencio, Meta, Colombia", "Granada, Meta, Colombia",
    "Montería, Córdoba, Colombia", "Cereté, Córdoba, Colombia", "Lorica, Córdoba, Colombia",
    "Bogotá, Colombia", "Medellín, Antioquia, Colombia", "Cali, Valle del Cauca, Colombia",
    "Barranquilla, Atlántico, Colombia", "Cartagena, Bolívar, Colombia", "Cúcuta, Norte de Santander, Colombia",
    "Bucaramanga, Santander, Colombia", "Pereira, Risaralda, Colombia", "Santa Marta, Magdalena, Colombia",
    "Ibagué, Tolima, Colombia", "Manizales, Caldas, Colombia", "Neiva, Huila, Colombia",
    "Pasto, Nariño, Colombia", "Armenia, Quindío, Colombia", "Sincelejo, Sucre, Colombia",
    "Popayán, Cauca, Colombia", "Valledupar, Cesar, Colombia", "Tunja, Boyacá, Colombia",
    "Yopal, Casanare, Colombia", "Girardot, Cundinamarca, Colombia", "Palmira, Valle del Cauca, Colombia",
    "Otra ciudad (escribir)…",
]

# ------------------------------------------------------------
# Utilidades de limpieza y parser (idénticas al pipeline validado)
# ------------------------------------------------------------
def norm(s):
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper()).strip()

TIPO_EJE = {"CALLE": "H", "CL": "H", "CLL": "H", "CALE": "H", "CALL": "H",
            "CARRERA": "V", "CARERA": "V", "CRA": "V", "CR": "V", "CRR": "V",
            "KR": "V", "KRA": "V", "K": "V",
            "AVENIDA": "AV", "AV": "AV", "AVE": "AV", "AVD": "AV",
            "DIAGONAL": "DG", "DG": "DG", "DIAG": "DG", "DIG": "DG",
            "TRANSVERSAL": "TV", "TV": "TV", "TRANS": "TV"}
PLURAL = {"CALLES": "CALLE", "CARRERAS": "CARRERA", "CRAS": "CRA", "CLLS": "CLL"}
TIPO_RE = ("CALLES|CALLE|CARRERAS|CARRERA|CARERA|AVENIDA|DIAGONAL|TRANSVERSAL|CALE|CALL|CLLS|CLL"
           "|CRAS|CRR|CRA|CR|KRA|KR|DIAG|DIG|DG|AVE|AVD|AV|TV|TRANS|CL|K")
NUMV = r"(\d+(?:\s?[A-JL-MO-Z](?![A-Z]))?)"
TOKEN = re.compile(rf"\b({TIPO_RE})\.?\s*{NUMV}")
CRUCE = re.compile(rf"(?:#|\bN[O.:\-]{{0,2}}\s*|\bNRO\.?\s*|\bCON\s+|\bENTRE\s+|\s){NUMV}\b")

def parsea(d):
    d = norm(d)
    if re.search(r"KILOMETRO|KM\s*\d|\d\s*KM|\bKM\b|VEREDA|FINCA|\bVIA\s+[A-Z]", d):
        return (None, None, "rural/km")
    toks = [(TIPO_EJE[PLURAL.get(m.group(1), m.group(1))], m.group(2).replace(" ", ""), m.end())
            for m in TOKEN.finditer(d)]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            if (toks[i][0], toks[i][1]) != (toks[j][0], toks[j][1]):
                return ((toks[i][0], toks[i][1]), (toks[j][0], toks[j][1]), "cruce explicito")
    if toks:
        t = toks[0]
        m = CRUCE.search(d, t[2])
        if m:
            num = m.group(1).replace(" ", "")
            eje = {"H": "V", "V": "H"}.get(t[0], "X")
            return ((t[0], t[1]), (eje, num), "numeracion/con/entre")
        return ((t[0], t[1]), None, "solo una via")
    return (None, None, "no parseable")

MAPA_GRAVEDAD = {"HERIDOS": "CON HERIDOS", "CON HERIDOS": "CON HERIDOS",
                 "MUERTOS": "CON MUERTOS", "CON MUERTOS": "CON MUERTOS",
                 "DANOS MATERIALES": "SOLO DANOS", "SOLO DANOS": "SOLO DANOS"}

COLS_ESPERADAS = ["fecha", "codigo", "direccion", "barrio", "vehiculos",
                  "heridos", "gravedad", "muertes", "clase"]

def limpia_csv(df):
    """Renombra por posición si el número de columnas coincide y normaliza."""
    if len(df.columns) == len(COLS_ESPERADAS):
        df = df.copy()
        df.columns = COLS_ESPERADAS
    else:
        faltan = set(COLS_ESPERADAS) - set(df.columns)
        if faltan:
            raise ValueError(f"El CSV debe tener 9 columnas o los nombres esperados. Faltan: {faltan}")
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y %b %d %I:%M:%S %p", errors="coerce")
    df["gravedad"] = df["gravedad"].map(norm).map(MAPA_GRAVEDAD)
    df["clase"] = df["clase"].map(norm)
    df["barrio"] = df["barrio"].map(norm).replace({"": "SIN INFORMACION"})
    df["muertes"] = pd.to_numeric(df["muertes"].replace("NO APLICA", 0), errors="coerce").fillna(0).astype(int)
    for c in ["heridos", "vehiculos"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["severidad"] = np.maximum(1, 10 * df["muertes"] + 3 * df["heridos"]
                                 + (df["gravedad"] == "SOLO DANOS").astype(int))
    df[["v1", "v2", "parse_metodo"]] = df["direccion"].apply(lambda x: pd.Series(parsea(x)))
    return df

# ------------------------------------------------------------
# Descarga de red y features (cacheadas: es la parte lenta)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def carga_red(ciudad, radio_km=0):
    """Descarga y proyecta el grafo de OSM. Si radio_km>0, descarga solo un
    círculo de ese radio alrededor del centro (mucho más rápido en ciudades
    grandes). Reintenta con servidores Overpass alternativos si el principal falla."""
    import osmnx as ox, time
    ox.settings.use_cache = True     # OSMnx cachea en disco la descarga de Overpass
    for attr, val in [("requests_timeout", 300), ("timeout", 300)]:
        try: setattr(ox.settings, attr, val)   # el nombre varía según versión de OSMnx
        except Exception: pass

    endpoints = ["https://overpass-api.de/api",
                 "https://overpass.kumi.systems/api",
                 "https://maps.mail.ru/osm/tools/overpass/api"]
    ultimo = None
    for ep in endpoints:
        for attr in ("overpass_url", "overpass_endpoint"):
            try: setattr(ox.settings, attr, ep)
            except Exception: pass
        for _ in range(2):                     # 2 intentos por servidor
            try:
                if radio_km and radio_km > 0:
                    lat, lon = ox.geocode(ciudad)
                    G = ox.graph_from_point((lat, lon), dist=int(radio_km * 1000),
                                            network_type="drive")
                else:
                    G = ox.graph_from_place(ciudad, network_type="drive")
                return G, ox.project_graph(G)  # UTM automático
            except Exception as e:
                ultimo = e
                time.sleep(3)
    raise ultimo

def clave_osm(nombre):
    m = TOKEN.match(norm(nombre))
    return (TIPO_EJE[PLURAL.get(m.group(1), m.group(1))], m.group(2).replace(" ", "")) if m else None

def candidatos(via):
    eje, num = via
    if eje in ("H", "V"): return [(eje, num)]
    if eje == "AV":       return [("AV", num), ("V", num), ("H", num)]
    if eje == "X":        return [("V", num), ("H", num)]
    return [(eje, num), ("V", num), ("H", num)]

@st.cache_resource(show_spinner=False)
def features_red(ciudad, radio_km=0):
    """Intersecciones + índices de vía + betweenness APROXIMADA, con caché en disco.
    El nº de pivotes de la betweenness se adapta al tamaño del grafo para no
    disparar el tiempo en ciudades grandes."""
    import osmnx as ox, networkx as nx

    ruta = os.path.join(CACHE_DIR, re.sub(r"\W+", "_", f"{ciudad}_r{radio_km}") + ".pkl")
    if os.path.exists(ruta):                 # ya se calculó antes -> lectura instantánea
        with open(ruta, "rb") as f:
            return pickle.load(f)

    G, G_proj = carga_red(ciudad, radio_km)
    nodos, aristas = ox.graph_to_gdfs(G_proj)
    via_a_nodos, via_a_aristas = {}, {}
    for (u, v, k), row in aristas.iterrows():
        nombres = row.get("name")
        if nombres is None or (isinstance(nombres, float) and pd.isna(nombres)):
            continue
        for nombre in (nombres if isinstance(nombres, list) else [nombres]):
            cl = clave_osm(nombre)
            if cl:
                via_a_nodos.setdefault(cl, set()).update([u, v])
                via_a_aristas.setdefault(cl, []).append((u, v, k))
    inter = nodos[nodos["street_count"] >= 3].copy()

    G_u = ox.convert.to_undirected(G_proj)
    n = G_u.number_of_nodes()
    # pivotes adaptativos: grafos grandes usan menos pivotes (ranking casi igual)
    k = 100 if n > 12000 else 250 if n > 5000 else min(500, n)
    btw = nx.betweenness_centrality(G_u, k=k, weight="length", seed=42)
    inter["grado"] = inter["street_count"]
    inter["betweenness"] = pd.Series(btw).reindex(inter.index).fillna(0)

    salida = (G, G_proj, nodos, aristas, inter, via_a_nodos, via_a_aristas)
    with open(ruta, "wb") as f:              # persiste para el próximo arranque
        pickle.dump(salida, f)
    return salida

CATEGORIAS_POI = {
    "educacion":   {"amenity": ["school", "university", "college", "kindergarten"]},
    "salud":       {"amenity": ["hospital", "clinic", "pharmacy", "doctors"]},
    "comercio":    {"shop": True},
    "gastronomia": {"amenity": ["restaurant", "cafe", "fast_food", "bar"]},
    "transporte":  {"amenity": ["bus_station", "taxi", "fuel"], "highway": ["bus_stop"]},
    "servicios":   {"amenity": ["bank", "police", "townhall", "post_office", "marketplace"]},
    "recreacion":  {"leisure": True, "amenity": ["place_of_worship"]},
}

@st.cache_resource(show_spinner=False)
def features_entorno(ciudad, radio_km=0):
    """Distancias y densidades de POIs por intersección. Si hay radio, los POIs
    también se descargan por radio (coherente y más rápido)."""
    import osmnx as ox
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN
    _, G_proj, nodos, _, inter, _, _ = features_red(ciudad, radio_km)
    centro_ll = ox.geocode(ciudad) if (radio_km and radio_km > 0) else None
    xy = np.c_[inter.geometry.x, inter.geometry.y]
    total = np.zeros(len(inter))
    out = {}
    for cat, tags in CATEGORIAS_POI.items():
        try:
            if centro_ll:
                g = ox.features_from_point(centro_ll, tags=tags,
                                           dist=int(radio_km * 1000)).to_crs(G_proj.graph["crs"])
            else:
                g = ox.features_from_place(ciudad, tags=tags).to_crs(G_proj.graph["crs"])
            pts = np.c_[g.geometry.centroid.x, g.geometry.centroid.y]
            lab = DBSCAN(eps=35, min_samples=1).fit_predict(pts)
            pts = np.array([pts[lab == l].mean(axis=0) for l in np.unique(lab)])
        except Exception:
            pts = np.empty((0, 2))
        if len(pts) == 0:
            out[f"dist_{cat}"], out[f"dens_{cat}_300m"] = np.full(len(inter), 3000.0), np.zeros(len(inter))
            continue
        t = cKDTree(pts)
        out[f"dist_{cat}"] = np.minimum(t.query(xy)[0], 3000)
        dens = np.array([len(t.query_ball_point(p, 300)) for p in xy])
        out[f"dens_{cat}_300m"] = dens
        total += dens
    out["generadores_300m"] = total
    out["dist_centro"] = np.linalg.norm(xy - xy.mean(axis=0), axis=1)
    return pd.DataFrame(out, index=inter.index)

def localiza(v1, v2, via_a_nodos, sc):
    # tras df.apply, los None se vuelven NaN (float); exigir tuplas reales
    if not isinstance(v1, tuple) or not isinstance(v2, tuple):
        return None
    for c1 in candidatos(v1):
        for c2 in candidatos(v2):
            cand = via_a_nodos.get(c1, set()) & via_a_nodos.get(c2, set())
            if cand:
                return max(cand, key=lambda n: sc.get(n, 0)), c1, c2
    return None

# ------------------------------------------------------------
# Métricas
# ------------------------------------------------------------
def recall_at_k(y, score, k=0.10):
    n = max(1, int(len(score) * k))
    return y[np.argsort(score)[-n:]].sum() / max(1, y.sum())

# ============================================================
# INTERFAZ
# ============================================================
st.title("🚦 Identificación de intersecciones y tramos viales críticos")
st.caption("Modo multi-ciudad · sube un CSV de accidentes e indica la ciudad. "
           "Basado en red vial (OSM), entorno urbano y validación cruzada espacial.")

with st.sidebar:
    st.header("⚙️ Configuración")

    # --- fuente de datos: CSV subido o datos generados en la otra página ---
    tiene_gen = "gen" in st.session_state
    opciones = ["Subir un CSV"] + (["Usar datos generados (página Generador)"] if tiene_gen else [])
    fuente = st.radio("Datos de accidentes", opciones)
    archivo = None
    if fuente == "Subir un CSV":
        archivo = st.file_uploader("CSV de accidentes", type=["csv"])
    else:
        st.success(f"Se usarán {len(st.session_state['gen'])} accidentes generados "
                   f"para «{st.session_state.get('gen_ciudad', '?')}».")

    # --- ciudad como desplegable buscable ---
    gc = st.session_state.get("gen_ciudad")
    lista = ([gc] if (gc and gc not in CIUDADES) else []) + CIUDADES
    if fuente != "Subir un CSV" and gc in lista:
        idx = lista.index(gc)                       # al usar datos generados, precarga su ciudad
    else:
        idx = lista.index("Acacías, Meta, Colombia") if "Acacías, Meta, Colombia" in lista else 0
    sel = st.selectbox("Ciudad (escribe para buscar)", lista, index=idx)
    if sel == "Otra ciudad (escribir)…":
        ciudad = st.text_input("Escribe la ciudad como en OpenStreetMap", "Acacías, Meta, Colombia")
    else:
        ciudad = sel

    # --- área a analizar (clave para ciudades grandes como Medellín) ---
    area = st.radio("Área a analizar", ["Centro urbano (rápido)", "Municipio completo (lento)"],
                    help="En ciudades grandes, analiza solo el centro por radio: mucho más rápido.")
    if area.startswith("Centro"):
        radio_km = st.slider("Radio del centro (km)", 2, 15, 6)
    else:
        radio_km = 0

    modelo_sel = st.selectbox("Modelo", ["Regresión Logística (interpretable)", "Random Forest"])
    bloque_km = st.slider("Tamaño de bloque espacial (km)", 0.5, 3.0, 1.0, 0.5,
                          help="Grupo de la validación cruzada espacial. Mayor = más estricto.")
    limitar_area = st.checkbox("Limitar análisis al área con datos", value=True,
                               help="Descarta intersecciones lejanas a cualquier accidente. Evita "
                                    "prioridades altas en zonas rurales sin siniestralidad.")
    buffer_m = st.slider("Radio de influencia de los datos (m)", 300, 3000, 1000, 100) if limitar_area else 0
    n_folds = st.slider("Número de pliegues (folds)", 3, 5, 5)
    ejecutar = st.button("▶️ Ejecutar análisis", type="primary")

    st.markdown("---")
    st.warning("**Rendimiento:** la primera ejecución de una ciudad descarga la red de OSM y "
               "calcula la centralidad (betweenness). Puede tardar **1–5 min** en ciudades "
               "pequeñas y **más en ciudades grandes**; Overpass ocasionalmente falla o limita "
               "peticiones (reintenta). Los resultados quedan **en caché**: reejecutar la misma "
               "ciudad es casi instantáneo. El módulo de visión satelital se omite por ser pesado "
               "y no haber mejorado el modelo.")

# ============================================================
# PIPELINE (se ejecuta SOLO al pulsar el botón; resultados en session_state
# para que las interacciones posteriores —radio de mapas, descargas— NO
# borren la vista ni recalculen todo).
# ============================================================
def mapa_tramos_severidad(loc, aristas_ll, via_a_aristas):
    """Tramos coloreados por severidad acumulada. LayerControl + leyenda de color."""
    import folium, branca
    centro = [aristas_ll.geometry.centroid.y.mean(), aristas_ll.geometry.centroid.x.mean()]
    sev_a, acc_a, via_a = {}, {}, {}
    for r in loc.itertuples():
        for clave in (r.clave_v1, r.clave_v2):
            if not isinstance(clave, tuple):
                continue
            for (u, v, k) in via_a_aristas.get(clave, []):
                if r.nodo in (u, v):
                    sev_a[(u, v, k)] = sev_a.get((u, v, k), 0) + r.severidad
                    acc_a[(u, v, k)] = acc_a.get((u, v, k), 0) + 1
                    via_a[(u, v, k)] = (f"{'Calle' if clave[0]=='H' else 'Carrera' if clave[0]=='V' else clave[0].title()}"
                                        f" {clave[1]}")
    tr = aristas_ll.reset_index()
    tr["severidad"]  = tr.apply(lambda r: sev_a.get((r.u, r.v, r.key), 0), axis=1)
    tr["accidentes"] = tr.apply(lambda r: acc_a.get((r.u, r.v, r.key), 0), axis=1)
    tr["via"]        = tr.apply(lambda r: via_a.get((r.u, r.v, r.key), ""), axis=1)
    tr = tr.to_crs(32618); tr["geometry"] = tr.geometry.simplify(8); tr = tr.to_crs(4326)
    m = folium.Map(centro, zoom_start=14, tiles="cartodbpositron", prefer_canvas=True)
    folium.GeoJson(tr[["geometry"]], style_function=lambda f: {"color": "#cccccc", "weight": 0.8},
                   name="Red vial").add_to(m)
    crit = tr[tr["accidentes"] > 0].copy()
    vmax = float(crit["severidad"].quantile(0.95)) if len(crit) else 1.0
    vmax = vmax or 1.0
    cmap = branca.colormap.linear.YlOrRd_09.scale(0, vmax)
    folium.GeoJson(crit[["geometry", "via", "accidentes", "severidad"]],
        style_function=lambda f: {"color": cmap(min(f["properties"]["severidad"], vmax)),
                                  "weight": 4.5, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=["via", "accidentes", "severidad"],
                                      aliases=["Vía", "Siniestros", "Severidad"]),
        name="Tramos con siniestros").add_to(m)
    cmap.caption = "Severidad del tramo"; m.add_child(cmap)
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

def mapa_intersecciones(loc, nodos_ll):
    """Círculos por severidad + calor + top 15 críticas. Capas conmutables."""
    import folium
    from folium.plugins import HeatMap
    centro = [nodos_ll.geometry.y.mean(), nodos_ll.geometry.x.mean()]
    agg_n = loc.groupby("nodo").agg(n_acc=("codigo", "count"), severidad=("severidad", "sum"),
                                    muertes=("muertes", "sum"), heridos=("heridos", "sum"))
    sitio  = loc.groupby("nodo")["direccion"].agg(lambda s: s.mode().iloc[0])
    barrio = loc.groupby("nodo")["barrio"].agg(lambda s: s.mode().iloc[0])
    def color(s):
        return "#b30000" if s >= 30 else "#e34a33" if s >= 15 else "#fc8d59" if s >= 6 else "#fdbb84"
    m = folium.Map(centro, zoom_start=14, tiles="cartodbpositron", prefer_canvas=True)
    fg = folium.FeatureGroup(name="Intersecciones (severidad)").add_to(m)
    for nodo, r in agg_n.iterrows():
        p = nodos_ll.loc[nodo].geometry
        folium.CircleMarker([float(p.y), float(p.x)], radius=float(3 + r["severidad"]**0.5),
            color=color(r["severidad"]), fill=True, fill_opacity=0.7, weight=0.5,
            popup=folium.Popup(f"<b>{sitio[nodo]}</b><br>{barrio[nodo]}<br>Siniestros: {int(r['n_acc'])}"
                               f"<br>Heridos: {int(r['heridos'])} | Muertes: {int(r['muertes'])}"
                               f"<br>Severidad: {int(r['severidad'])}", max_width=250)).add_to(fg)
    HeatMap([[float(nodos_ll.loc[n].geometry.y), float(nodos_ll.loc[n].geometry.x), float(r["severidad"])]
             for n, r in agg_n.iterrows()], radius=18, blur=14, name="Calor de severidad").add_to(m)
    fgt = folium.FeatureGroup(name="Top 15 críticas", show=False).add_to(m)
    for nodo, r in agg_n.nlargest(15, "severidad").iterrows():
        p = nodos_ll.loc[nodo].geometry
        folium.Marker([float(p.y), float(p.x)], icon=folium.Icon(color="red", icon="exclamation-sign"),
            popup=folium.Popup(f"<b>{sitio[nodo]}</b><br>Siniestros: {int(r['n_acc'])}"
                               f"<br>Heridos: {int(r['heridos'])} | Muertes: {int(r['muertes'])}"
                               f"<br>Severidad: {int(r['severidad'])}", max_width=250)).add_to(fgt)
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

def mapa_prioridad(inter, aristas_ll):
    """Tramos en 5 clases de prioridad (percentil) + inspección/prevención.
    LayerControl con casillas por clase + leyenda HTML fija."""
    import folium
    p_nodo = inter["pct_p"].to_dict()
    top_resid = set(inter[inter["y"] == 1].nlargest(12, "residuo").index)
    top_prev  = set(inter[inter["y"] == 0].nlargest(12, "p_oof").index)
    tr = aristas_ll.reset_index()
    def p_de(u, v):
        vals = [p_nodo[n] for n in (u, v) if n in p_nodo]
        return float(np.mean(vals)) if vals else np.nan
    tr["p_tramo"]  = tr.apply(lambda r: p_de(r.u, r.v), axis=1)
    tr["alerta"]   = tr.apply(lambda r: (r.u in top_resid) or (r.v in top_resid), axis=1)
    tr["prevenir"] = tr.apply(lambda r: (r.u in top_prev) or (r.v in top_prev), axis=1)
    tr["via"] = tr["name"].apply(lambda n: (n[0] if isinstance(n, list) else n) or "")
    tr = tr.to_crs(32618); tr["geometry"] = tr.geometry.simplify(8); tr = tr.to_crs(4326)
    def clase(p):
        if p >= 0.95: return "Muy alta (top 5%)"
        if p >= 0.90: return "Alta (top 10%)"
        if p >= 0.75: return "Media (top 25%)"
        if p >= 0.50: return "Baja"
        return "Mínima"
    ESTILO = {"Muy alta (top 5%)": ("#a50026", 6.0), "Alta (top 10%)": ("#e34a33", 4.5),
              "Media (top 25%)": ("#fdae61", 3.0), "Baja": ("#ffe8a8", 1.8), "Mínima": ("#d9d9d9", 1.0)}
    tr["clase"] = tr["p_tramo"].apply(lambda p: clase(p) if pd.notna(p) else "Mínima")
    centro = [aristas_ll.geometry.centroid.y.mean(), aristas_ll.geometry.centroid.x.mean()]
    m = folium.Map(centro, zoom_start=14, tiles="cartodbpositron", prefer_canvas=True)
    for nombre in ["Mínima", "Baja", "Media (top 25%)", "Alta (top 10%)", "Muy alta (top 5%)"]:
        sub = tr[tr["clase"] == nombre]
        if len(sub) == 0:
            continue
        col, w = ESTILO[nombre]
        folium.GeoJson(sub[["geometry", "via", "p_tramo"]].assign(p_tramo=lambda d: d["p_tramo"].round(3)),
            style_function=lambda f, c=col, ww=w: {"color": c, "weight": ww, "opacity": 0.9},
            tooltip=folium.GeoJsonTooltip(fields=["via", "p_tramo"], aliases=["Vía", "Percentil"]),
            name=f"Prioridad {nombre}").add_to(m)
    folium.GeoJson(tr[tr["alerta"]][["geometry", "via"]],
        style_function=lambda f: {"color": "#7b3294", "weight": 6, "opacity": 0.95, "dashArray": "6 4"},
        tooltip=folium.GeoJsonTooltip(fields=["via"], aliases=["Inspección de campo:"]),
        name="Más peligrosos de lo esperado (inspección)", show=True).add_to(m)
    folium.GeoJson(tr[tr["prevenir"] & ~tr["alerta"]][["geometry", "via"]],
        style_function=lambda f: {"color": "#0571b0", "weight": 4.5, "opacity": 0.95, "dashArray": "2 6"},
        tooltip=folium.GeoJsonTooltip(fields=["via"], aliases=["Prevención:"]),
        name="Riesgo estructural (prevención)", show=False).add_to(m)
    leyenda = """<div style="position:fixed; bottom:24px; left:12px; z-index:9999; background:white;
     padding:10px 14px; border-radius:6px; box-shadow:0 1px 5px rgba(0,0,0,.3); font-size:12px;">
     <b>Prioridad de intervención</b><br>
     <span style="color:#a50026;">━━</span> Muy alta (top 5%)<br>
     <span style="color:#e34a33;">━━</span> Alta (top 10%)<br>
     <span style="color:#fdae61;">━━</span> Media (top 25%)<br>
     <span style="color:#ffe8a8;">━━</span> Baja<br>
     <span style="color:#d9d9d9;">━━</span> Mínima</div>"""
    m.get_root().html.add_child(folium.Element(leyenda))
    folium.LayerControl(collapsed=False).add_to(m)
    return m.get_root().render()

if ejecutar:
    if fuente == "Subir un CSV" and archivo is None:
        st.error("Sube un archivo CSV para continuar.")
        st.stop()
    from scipy.spatial import cKDTree
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                                 roc_curve, precision_recall_curve)
    import osmnx as ox

    avisos = []
    try:
        if fuente == "Subir un CSV":
            df = limpia_csv(pd.read_csv(archivo))
        else:
            df = limpia_csv(st.session_state["gen"].copy())   # datos de la página Generador
    except Exception as e:
        st.error(f"Error leyendo/limpiando los datos: {e}"); st.stop()

    urb = df["parse_metodo"] != "rural/km"
    comp = df["v1"].notna() & df["v2"].notna()

    try:
        with st.spinner(f"Descargando red vial y calculando centralidad de «{ciudad}»… (puede tardar)"):
            G, G_proj, nodos, aristas, inter, via_a_nodos, via_a_aristas = features_red(ciudad, radio_km)
            ent = features_entorno(ciudad, radio_km)
    except Exception as e:
        st.error(f"No se pudo construir la red de «{ciudad}». Verifica el nombre en OSM o reintenta "
                 f"(Overpass puede estar saturado). Detalle: {e}"); st.stop()

    inter = inter.join(ent)

    with st.spinner("Geocodificando, construyendo características y entrenando el modelo…"):
        sc = pd.Series(dict(G_proj.nodes(data="street_count")))
        res = df.apply(lambda r: localiza(r["v1"], r["v2"], via_a_nodos, sc), axis=1)
        df["nodo"]     = res.map(lambda t: t[0] if t else None)
        df["clave_v1"] = res.map(lambda t: t[1] if t else None)
        df["clave_v2"] = res.map(lambda t: t[2] if t else None)

        xy_inter = np.c_[inter.geometry.x, inter.geometry.y]
        tree_i = cKDTree(xy_inter)
        loc = df.dropna(subset=["nodo"]).copy()
        if len(loc) == 0:
            st.error("Ningún accidente pudo ubicarse en la red de esta ciudad. ¿La nomenclatura del "
                     "CSV corresponde a la ciudad indicada?"); st.stop()
        xy_acc = np.c_[nodos.loc[loc["nodo"]].geometry.x, nodos.loc[loc["nodo"]].geometry.y]
        d, ix = tree_i.query(xy_acc)
        loc["inter_id"] = np.where(d <= 150, inter.index[ix], -1)
        agg = (loc[loc["inter_id"] != -1].groupby("inter_id")
               .agg(n_acc=("codigo", "count"), severidad=("severidad", "sum"),
                    muertes=("muertes", "sum"), heridos=("heridos", "sum")))
        inter = inter.join(agg).fillna({"n_acc": 0, "severidad": 0, "muertes": 0, "heridos": 0})
        inter["y"] = (inter["n_acc"] > 0).astype(int)

        # limitar a intersecciones cercanas a algún accidente (las con siniestro
        # tienen distancia 0 y siempre se conservan). Elimina las intersecciones
        # rurales lejanas donde el modelo marcaría prioridad sin datos que lo respalden.
        if limitar_area and len(loc):
            acc_xy = np.c_[nodos.loc[loc["nodo"]].geometry.x, nodos.loc[loc["nodo"]].geometry.y]
            dnear = cKDTree(acc_xy).query(np.c_[inter.geometry.x, inter.geometry.y])[0]
            inter = inter[dnear <= buffer_m].copy()

        inter["bloque"] = (np.floor(inter.geometry.x / (bloque_km*1000)).astype(int).astype(str) + "_"
                           + np.floor(inter.geometry.y / (bloque_km*1000)).astype(int).astype(str))

        FEAT = (["grado", "betweenness", "generadores_300m", "dist_centro"]
                + [f"dist_{c}" for c in CATEGORIAS_POI] + [f"dens_{c}_300m" for c in CATEGORIAS_POI])
        X = inter[FEAT].fillna(inter[FEAT].median())
        y = inter["y"].values
        grupos = inter["bloque"].values

        def crea_modelo():
            if modelo_sel.startswith("Regresión"):
                return make_pipeline(StandardScaler(),
                                     LogisticRegression(class_weight="balanced", max_iter=3000))
            return RandomForestClassifier(400, min_samples_leaf=5, class_weight="balanced",
                                          random_state=42, n_jobs=-1)

        if y.sum() < n_folds or (y == 0).sum() < n_folds:
            avisos.append(f"Pocos casos positivos ({int(y.sum())}) para {n_folds} pliegues; "
                          f"la validación puede ser inestable.")
        oof = np.zeros(len(y))
        n_ok = min(n_folds, pd.Series(grupos).nunique())
        try:
            for tr, te in GroupKFold(n_splits=n_ok).split(X, y, grupos):
                m = crea_modelo(); m.fit(X.iloc[tr], y[tr])
                oof[te] = m.predict_proba(X.iloc[te])[:, 1]
            metrico = True
        except Exception as e:
            avisos.append(f"No se pudo completar la CV espacial ({e}); ajuste completo sin métricas OOF.")
            m = crea_modelo(); m.fit(X, y); oof = m.predict_proba(X)[:, 1]; metrico = False

        inter["p_oof"] = oof
        inter["residuo"] = y - oof
        inter["pct_p"] = pd.Series(oof, index=inter.index).rank(pct=True)

        nodos_ll, aristas_ll = ox.graph_to_gdfs(G)
        ref = loc[loc["inter_id"] != -1].groupby("inter_id")["direccion"].agg(lambda s: s.mode().iloc[0])
        metr = {}
        if metrico:
            cm = confusion_matrix(y, (oof >= np.quantile(oof, 0.90)).astype(int))
            fpr, tpr, _ = roc_curve(y, oof)
            prec, rec, _ = precision_recall_curve(y, oof)
            metr = {"roc": roc_auc_score(y, oof), "pr": average_precision_score(y, oof),
                    "rec": recall_at_k(y, oof), "cm": cm,
                    "fpr": fpr, "tpr": tpr, "prec": prec, "rec_curve": rec}
        # guardar TODO lo necesario para mostrar sin recalcular
        st.session_state["R"] = {
            "df": df, "inter": inter, "loc": loc, "ref": ref, "y": y, "oof": oof,
            "metrico": metrico, "metr": metr, "avisos": avisos,
            "cobertura": comp.sum()/max(1, urb.sum()), "rurales": int((~urb).sum()),
            "prevalencia": float(y.mean()), "ubicados": int(df["nodo"].notna().sum()),
            "mapa_severidad": mapa_tramos_severidad(loc, aristas_ll, via_a_aristas),
            "mapa_int": mapa_intersecciones(loc, nodos_ll),
            "mapa_prioridad": mapa_prioridad(inter, aristas_ll),
        }

# ============================================================
# VISUALIZACIÓN (lee de session_state; sobrevive a reruns)
# ============================================================
if "R" not in st.session_state:
    st.info("Configura la ciudad y sube el CSV en la barra lateral, luego pulsa **Ejecutar análisis**. "
            "El CSV debe tener las columnas: fecha, código, dirección, barrio, vehículos, heridos, "
            "gravedad (con heridos/con muertos/solo daños), muertes y clase.")
    st.stop()

import plotly.express as px
import streamlit.components.v1 as components
R = st.session_state["R"]
df, inter, loc = R["df"], R["inter"], R["loc"]
y, oof = R["y"], R["oof"]
for a in R["avisos"]:
    st.warning(a)

t1, t2, t3, t4 = st.tabs(["📊 Análisis descriptivo", "🧭 Geocodificación", "🤖 Modelo y métricas", "🗺️ Mapas"])

with t1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidentes", len(df))
    c2.metric("Urbanos ubicados", R["ubicados"])
    c3.metric("Intersecciones", len(inter))
    c4.metric("Críticas (≥1 sin.)", int(y.sum()), f"{R['prevalencia']:.1%} prevalencia")

    dfm = df.dropna(subset=["fecha"]).copy()
    dfm["anio"] = dfm["fecha"].dt.year
    dfm["mes"] = dfm["fecha"].dt.month
    dfm["dia_sem"] = dfm["fecha"].dt.day_name()

    # 1) serie temporal (área interactiva)
    serie = dfm.set_index("fecha").resample("ME").size().reset_index(name="accidentes")
    fig = px.area(serie, x="fecha", y="accidentes", title="Siniestros por mes",
                  color_discrete_sequence=["#d7301f"])
    fig.update_layout(height=320, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    cc = st.columns(2)
    # 2) heatmap año × mes
    piv = (dfm.pivot_table(index="anio", columns="mes", values="codigo", aggfunc="count")
           .reindex(columns=range(1, 13)))
    fig2 = px.imshow(piv, color_continuous_scale="YlOrRd", aspect="auto",
                     labels=dict(color="casos"), title="Estacionalidad: año × mes")
    fig2.update_layout(height=320, margin=dict(t=40, b=0))
    cc[0].plotly_chart(fig2, use_container_width=True)
    # 3) donut de gravedad
    g = df["gravedad"].value_counts().reset_index()
    g.columns = ["gravedad", "n"]
    fig3 = px.pie(g, names="gravedad", values="n", hole=0.45, title="Distribución por gravedad",
                  color_discrete_sequence=px.colors.sequential.YlOrRd_r)
    fig3.update_layout(height=320, margin=dict(t=40, b=0))
    cc[1].plotly_chart(fig3, use_container_width=True)

    cc2 = st.columns(2)
    # 4) barras por clase
    cl = df["clase"].value_counts().reset_index(); cl.columns = ["clase", "n"]
    fig4 = px.bar(cl, x="n", y="clase", orientation="h", title="Clase de accidente",
                  color="n", color_continuous_scale="YlOrRd")
    fig4.update_layout(height=340, margin=dict(t=40, b=0), showlegend=False,
                       yaxis={"categoryorder": "total ascending"})
    cc2[0].plotly_chart(fig4, use_container_width=True)
    # 5) día de la semana
    orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    esp = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    dw = dfm["dia_sem"].value_counts().reindex(orden).fillna(0).reset_index()
    dw.columns = ["dia", "n"]; dw["dia"] = esp
    fig5 = px.bar(dw, x="dia", y="n", title="Siniestros por día de la semana",
                  color="n", color_continuous_scale="YlOrRd")
    fig5.update_layout(height=340, margin=dict(t=40, b=0), showlegend=False)
    cc2[1].plotly_chart(fig5, use_container_width=True)

    # 6) treemap de barrios
    b = df[df["barrio"] != "SIN INFORMACION"]["barrio"].value_counts().head(20).reset_index()
    b.columns = ["barrio", "n"]
    fig6 = px.treemap(b, path=["barrio"], values="n", title="Concentración por barrio (top 20)",
                      color="n", color_continuous_scale="YlOrRd")
    fig6.update_layout(height=380, margin=dict(t=40, b=0))
    st.plotly_chart(fig6, use_container_width=True)

with t2:
    st.write(f"**Cobertura del parser** (urbanas): {R['cobertura']:.1%} · "
             f"rurales excluidas: {R['rurales']}")
    vc = df["parse_metodo"].value_counts().reset_index()
    vc.columns = ["método", "conteo"]
    fig = px.bar(vc, x="conteo", y="método", orientation="h", color="conteo",
                 color_continuous_scale="Blues", title="Métodos de geocodificación")
    fig.update_layout(height=300, margin=dict(t=40, b=0), showlegend=False,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Las direcciones no resolubles se dejan sin ubicar: la app no inventa coordenadas.")

with t3:
    if R["metrico"]:
        M = R["metr"]
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{M['roc']:.3f}")
        c2.metric("PR-AUC", f"{M['pr']:.3f}", f"base {R['prevalencia']:.3f}")
        c3.metric("Recall@10%", f"{M['rec']:.3f}")

        cc = st.columns(2)
        # curva ROC
        figroc = px.area(x=M["fpr"], y=M["tpr"],
                         labels={"x": "Tasa de falsos positivos (FPR)", "y": "Tasa de verdaderos positivos (TPR)"},
                         title=f"Curva ROC (AUC = {M['roc']:.3f})",
                         color_discrete_sequence=["#d7301f"])
        figroc.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, y0=0, x1=1, y1=1)
        figroc.update_layout(height=360, margin=dict(t=40, b=0), xaxis_range=[0, 1], yaxis_range=[0, 1])
        cc[0].plotly_chart(figroc, use_container_width=True)
        # curva Precisión-Exhaustividad
        figpr = px.line(x=M["rec_curve"], y=M["prec"],
                        labels={"x": "Exhaustividad (Recall)", "y": "Precisión"},
                        title=f"Curva Precisión-Recall (PR-AUC = {M['pr']:.3f})",
                        color_discrete_sequence=["#2b8cbe"])
        figpr.add_hline(y=R["prevalencia"], line_dash="dash", line_color="gray",
                        annotation_text=f"base = {R['prevalencia']:.3f}")
        figpr.update_layout(height=360, margin=dict(t=40, b=0), xaxis_range=[0, 1], yaxis_range=[0, 1])
        cc[1].plotly_chart(figpr, use_container_width=True)

        cm = M["cm"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: no", "Pred: sí"], y=["Real: no", "Real: sí"],
                        title="Matriz de confusión (umbral top 10%)")
        fig.update_layout(height=340, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Predicciones out-of-fold con validación cruzada espacial por bloques. "
                   "Con prevalencia baja se priorizan ROC-AUC, PR-AUC (vs prevalencia) y Recall@10%.")
    st.subheader("Más peligrosas de lo esperado (inspección de campo)")
    tt = inter.join(R["ref"])
    st.dataframe(tt[tt["y"] == 1].nlargest(10, "residuo")
                 [["direccion", "n_acc", "severidad", "p_oof", "residuo"]].round(3))
    st.subheader("Riesgo estructural sin siniestros (prevención)")
    st.dataframe(inter[inter["y"] == 0].nlargest(10, "p_oof")
                 [["n_acc", "p_oof", "grado", "betweenness", "generadores_300m"]].round(3))

with t4:
    st.caption("Usa el control de capas (arriba a la derecha de cada mapa) para activar y "
               "desactivar capas.")
    vista = st.radio("Vista", ["Tramos críticos (severidad)", "Intersecciones críticas",
                               "Vías por prioridad predicha"], horizontal=True)
    if vista.startswith("Tramos"):
        html = R["mapa_severidad"]
    elif vista.startswith("Intersecciones"):
        html = R["mapa_int"]
    else:
        html = R["mapa_prioridad"]
    components.html(html, height=600, scrolling=False)

# ---------- Descargas ----------
st.sidebar.markdown("---")
salida = inter.drop(columns="geometry").copy()
st.sidebar.download_button("⬇️ Descargar intersecciones con predicciones (CSV)",
                           salida.to_csv().encode("utf-8"),
                           file_name="intersecciones_predicciones.csv", mime="text/csv")



