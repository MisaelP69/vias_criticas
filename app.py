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
def carga_red(ciudad):
    """Descarga y proyecta el grafo de OSM. Cacheado por ciudad."""
    import osmnx as ox
    ox.settings.use_cache = True     # OSMnx cachea en disco la descarga de Overpass
    ox.settings.timeout = 180
    G = ox.graph_from_place(ciudad, network_type="drive")
    G_proj = ox.project_graph(G)                       # UTM automático
    return G, G_proj

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
def features_red(ciudad):
    """Intersecciones + índices de vía + betweenness APROXIMADA, con caché en disco.
    Optimizaciones: (1) betweenness por muestreo de pivotes (k) -> mucho más rápida
    con ordenamiento casi idéntico; (2) persistencia en disco -> reinicios instantáneos."""
    import osmnx as ox, networkx as nx

    ruta = os.path.join(CACHE_DIR, re.sub(r"\W+", "_", ciudad) + ".pkl")
    if os.path.exists(ruta):                 # ya se calculó antes -> lectura instantánea
        with open(ruta, "rb") as f:
            return pickle.load(f)

    G, G_proj = carga_red(ciudad)
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
    k = min(500, G_u.number_of_nodes())      # muestreo: 500 pivotes (exacta si es menor)
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
def features_entorno(ciudad):
    """Distancias y densidades de POIs por intersección. Cacheado por ciudad."""
    import osmnx as ox
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN
    _, G_proj, nodos, _, inter, _, _ = features_red(ciudad)
    xy = np.c_[inter.geometry.x, inter.geometry.y]
    total = np.zeros(len(inter))
    out = {}
    for cat, tags in CATEGORIAS_POI.items():
        try:
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
    ciudad = st.text_input("Ciudad (como en OpenStreetMap)", "Acacías, Meta, Colombia",
                           help="Ej: 'Villavicencio, Meta, Colombia'. Debe existir en OSM.")
    archivo = st.file_uploader("CSV de accidentes", type=["csv"])
    modelo_sel = st.selectbox("Modelo", ["Regresión Logística (interpretable)", "Random Forest"])
    bloque_km = st.slider("Tamaño de bloque espacial (km)", 0.5, 3.0, 1.0, 0.5,
                          help="Grupo de la validación cruzada espacial. Mayor = más estricto.")
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
def construye_mapa_intersecciones(inter, nodos_ll):
    import folium
    from folium.plugins import HeatMap
    centro = [nodos_ll.geometry.y.mean(), nodos_ll.geometry.x.mean()]
    m = folium.Map(centro, zoom_start=14, tiles="cartodbpositron", prefer_canvas=True)
    agg_n = inter[inter["n_acc"] > 0]
    if len(agg_n):
        imax = float(agg_n["severidad"].max())
        HeatMap([[float(nodos_ll.loc[n].geometry.y), float(nodos_ll.loc[n].geometry.x),
                  float(s / imax)] for n, s in agg_n["severidad"].items()],
                radius=18, blur=14).add_to(m)
        for n, r in agg_n.nlargest(15, "severidad").iterrows():
            p = nodos_ll.loc[n].geometry
            folium.CircleMarker([float(p.y), float(p.x)], radius=float(3 + r["severidad"]**0.5),
                color="#b30000", fill=True, fill_opacity=0.7, weight=0.5,
                tooltip=f"Siniestros: {int(r['n_acc'])} · Severidad: {int(r['severidad'])}").add_to(m)
    return m.get_root().render()

def construye_mapa_tramos(inter, aristas_ll):
    import folium, branca
    centro = [aristas_ll.geometry.centroid.y.mean(), aristas_ll.geometry.centroid.x.mean()]
    m = folium.Map(centro, zoom_start=14, tiles="cartodbpositron", prefer_canvas=True)
    p_nodo = inter["pct_p"].to_dict()
    tr = aristas_ll.reset_index()
    def p_de(u, v):
        vals = [p_nodo[n] for n in (u, v) if n in p_nodo]
        return float(np.mean(vals)) if vals else np.nan
    tr["p_tramo"] = tr.apply(lambda r: p_de(r.u, r.v), axis=1)
    tr["via"] = tr["name"].apply(lambda n: (n[0] if isinstance(n, list) else n) or "")
    folium.GeoJson(tr[tr["p_tramo"].isna()][["geometry"]],
                   style_function=lambda f: {"color": "#d9d9d9", "weight": 0.8}).add_to(m)
    cmap = branca.colormap.linear.YlOrRd_09.scale(0, 1)
    con_p = tr[tr["p_tramo"].notna()].copy()
    con_p["p_tramo"] = con_p["p_tramo"].round(3)
    folium.GeoJson(con_p[["geometry", "via", "p_tramo"]],
        style_function=lambda f: {"color": cmap(f["properties"]["p_tramo"]),
                                  "weight": 1.5 + 5 * f["properties"]["p_tramo"], "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=["via", "p_tramo"],
                                      aliases=["Vía", "Percentil criticidad"])).add_to(m)
    cmap.caption = "Percentil de criticidad predicha del tramo"; m.add_child(cmap)
    return m.get_root().render()

if ejecutar:
    if archivo is None:
        st.error("Sube un archivo CSV para continuar.")
        st.stop()
    from scipy.spatial import cKDTree
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    import osmnx as ox

    avisos = []
    try:
        df = limpia_csv(pd.read_csv(archivo))
    except Exception as e:
        st.error(f"Error leyendo/limpiando el CSV: {e}"); st.stop()

    urb = df["parse_metodo"] != "rural/km"
    comp = df["v1"].notna() & df["v2"].notna()

    try:
        with st.spinner(f"Descargando red vial y calculando centralidad de «{ciudad}»… (puede tardar)"):
            G, G_proj, nodos, aristas, inter, via_a_nodos, via_a_aristas = features_red(ciudad)
            ent = features_entorno(ciudad)
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
            metr = {"roc": roc_auc_score(y, oof), "pr": average_precision_score(y, oof),
                    "rec": recall_at_k(y, oof), "cm": cm}
        # guardar TODO lo necesario para mostrar sin recalcular
        st.session_state["R"] = {
            "df": df, "inter": inter, "loc": loc, "ref": ref, "y": y, "oof": oof,
            "metrico": metrico, "metr": metr, "avisos": avisos,
            "cobertura": comp.sum()/max(1, urb.sum()), "rurales": int((~urb).sum()),
            "prevalencia": float(y.mean()), "ubicados": int(df["nodo"].notna().sum()),
            "mapa_int": construye_mapa_intersecciones(inter, nodos_ll),
            "mapa_tramos": construye_mapa_tramos(inter, aristas_ll),
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
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{R['metr']['roc']:.3f}")
        c2.metric("PR-AUC", f"{R['metr']['pr']:.3f}", f"base {R['prevalencia']:.3f}")
        c3.metric("Recall@10%", f"{R['metr']['rec']:.3f}")
        cm = R["metr"]["cm"]
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
    vista = st.radio("Vista", ["Intersecciones (severidad)", "Tramos por criticidad predicha"],
                     horizontal=True)
    html = R["mapa_int"] if vista.startswith("Intersecciones") else R["mapa_tramos"]
    components.html(html, height=580, scrolling=False)

# ---------- Descargas ----------
st.sidebar.markdown("---")
salida = inter.drop(columns="geometry").copy()
st.sidebar.download_button("⬇️ Descargar intersecciones con predicciones (CSV)",
                           salida.to_csv().encode("utf-8"),
                           file_name="intersecciones_predicciones.csv", mime="text/csv")
