# ============================================================
# PÁGINA 2: Generador de datos simulados (parametrizable).
# COLOCA ESTE ARCHIVO DENTRO DE UNA CARPETA LLAMADA  pages/
# es decir, en el repositorio la ruta debe ser:  pages/1_Generador_de_datos.py
# Streamlit lo mostrará automáticamente en el menú lateral, como segunda
# página junto a la principal (app.py). Usa claves de session_state propias
# ("gen") que no chocan con las de la app de análisis ("R").
# ============================================================
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Generador de datos", page_icon="🧪", layout="wide")
st.title("🧪 Generador de accidentes simulados")
st.caption("Configura todo a tu gusto y descarga un CSV listo para la página de análisis.")

# ------------------------------------------------------------
# CONTROLES
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros")
    ciudad = st.text_input("Ciudad (solo para el nombre del archivo)", "Montería, Córdoba, Colombia")
    n_total = st.number_input("Cantidad de accidentes", 50, 20000, 1000, step=50)
    col = st.columns(2)
    anio_min = col[0].number_input("Año inicial", 2010, 2035, 2021)
    anio_max = col[1].number_input("Año final", 2010, 2035, 2025)

    st.subheader("Rango de la malla vial")
    col = st.columns(2)
    calle_min = col[0].number_input("Calle mínima", 1, 300, 18)
    calle_max = col[1].number_input("Calle máxima", 1, 300, 63)
    col = st.columns(2)
    cra_min = col[0].number_input("Carrera mínima", 1, 300, 1)
    cra_max = col[1].number_input("Carrera máxima", 1, 300, 16)

    st.subheader("Puntos críticos (hotspots)")
    n_hot = st.slider("Número de intersecciones hotspot", 0, 30, 12)
    hot_min, hot_max = st.slider("Accidentes por hotspot (rango)", 1, 80, (20, 40))

    st.subheader("Mezcla de gravedad (se normaliza)")
    p_her = st.slider("% con heridos", 0.0, 1.0, 0.72, 0.01)
    p_dan = st.slider("% solo daños", 0.0, 1.0, 0.20, 0.01)
    p_mue = st.slider("% con muertos", 0.0, 1.0, 0.08, 0.01)

    st.subheader("Otros")
    p_rural = st.slider("% accidentes rurales (por km, se excluyen en la app)", 0.0, 0.3, 0.06, 0.01)
    barrios_txt = st.text_area("Barrios (uno por línea)",
        "CENTRO\nLA CEIBA\nSUCRE\nBUENAVISTA\nMONTERIA MODERNO\nLOS ALCAZARES\nEL RECREO\n"
        "SANTA FE\nNARINO\nLA GRANJA\nP-5\nGALILEA\nEL DORADO\nCANTACLARO\nMOGAMBO\nSIN INFORMACION",
        height=120)
    vias_rurales_txt = st.text_input("Vías rurales (separadas por coma)",
                                     "MONTERIA-CERETE, MONTERIA-PLANETA RICA, MONTERIA-ARBOLETES")
    semilla = st.number_input("Semilla aleatoria", 0, 999999, 2026)
    generar = st.button("▶️ Generar datos", type="primary")

# ------------------------------------------------------------
# GENERACIÓN
# ------------------------------------------------------------
def genera(params):
    rng = np.random.default_rng(params["semilla"])
    barrios = [b.strip().upper() for b in params["barrios"].splitlines() if b.strip()] or ["SIN INFORMACION"]
    vias_rur = [v.strip().upper() for v in params["vias_rurales"].split(",") if v.strip()] or ["VIA RURAL"]
    clases = ["CHOQUE", "Choque", "ATROPELLO", "Atropello", "VOLCAMIENTO", "OTRO"]

    ps = np.array([params["p_her"], params["p_dan"], params["p_mue"]], dtype=float)
    ps = ps / ps.sum() if ps.sum() > 0 else np.array([0.72, 0.20, 0.08])
    grav_labels = ["CON HERIDOS", "SOLO DANOS", "CON MUERTOS"]

    d0 = dt.date(int(params["anio_min"]), 1, 1)
    dias = max(1, (dt.date(int(params["anio_max"]), 12, 31) - d0).days)

    def fecha():
        return (d0 + dt.timedelta(days=int(rng.integers(0, dias + 1)))).strftime("%Y %b %d 12:00:00 AM")

    def direccion(c, k):
        e = rng.integers(0, 4)
        if e == 0: return f"Calle {c} con Cra {k}"
        if e == 1: return f"Calle {c} Carrera {k}"
        if e == 2: return f"Cll {c} N {k}-{rng.integers(10, 90)}"
        return f"Carrera {k} con Calle {c}"

    filas = []
    cod = 300000
    def add(c, k, barrio, grave):
        nonlocal cod
        cod += 1
        if grave == "CON MUERTOS":
            heridos = int(rng.integers(0, 3)); mtxt = str(int(rng.integers(1, 4)))
            gtxt = rng.choice(["MUERTOS", "Con Muertos"])
        elif grave == "SOLO DANOS":
            heridos = 0; mtxt = rng.choice(["NO APLICA", "0"]); gtxt = rng.choice(["DAÑOS MATERIALES", "Solo Daños"])
        else:
            heridos = int(rng.integers(1, 6)); mtxt = rng.choice(["NO APLICA", "0"]); gtxt = rng.choice(["HERIDOS", "Con Heridos"])
        veh = int(rng.choice([1, 2, 3, 4], p=[0.16, 0.62, 0.16, 0.06]))
        filas.append([fecha(), f"{cod:08d}", direccion(c, k), barrio, veh, heridos, gtxt, mtxt, rng.choice(clases)])

    n_total = int(params["n_total"])
    n_rural = int(n_total * params["p_rural"])
    n_urb = n_total - n_rural

    hotspots = []
    for _ in range(int(params["n_hot"])):
        hotspots.append((int(rng.integers(params["calle_min"], params["calle_max"] + 1)),
                         int(rng.integers(params["cra_min"], params["cra_max"] + 1)),
                         rng.choice(barrios)))
    for c, k, barrio in hotspots:
        for _ in range(int(rng.integers(params["hot_min"], params["hot_max"] + 1))):
            if len(filas) >= n_urb: break
            add(c, k, barrio, rng.choice(grav_labels, p=[0.60, 0.22, 0.18]))

    while len(filas) < n_urb:
        c = int(rng.integers(params["calle_min"], params["calle_max"] + 1))
        k = int(rng.integers(params["cra_min"], params["cra_max"] + 1))
        add(c, k, rng.choice(barrios), rng.choice(grav_labels, p=ps))

    for _ in range(n_rural):
        cod += 1
        km = rng.integers(1, 70)
        filas.append([fecha(), f"{cod:08d}",
                      f"KILOMETRO {km}+{rng.integers(100, 900)} MTS VIA {rng.choice(vias_rur)}",
                      "SIN INFORMACION", int(rng.choice([1, 2, 3])), int(rng.integers(0, 5)),
                      rng.choice(["HERIDOS", "MUERTOS", "DAÑOS MATERIALES"]),
                      rng.choice(["NO APLICA", "1", "2", "0"]), rng.choice(clases)])

    df = pd.DataFrame(filas, columns=["Fecha_Ocurrencia", "Codigo_Accidente", "Direccion", "Barrio",
        "Vehiculos Involucrados", "Heridos", "Accidente con", "Muertes", "Clase de Accidente"])
    return df.sample(frac=1, random_state=int(params["semilla"])).reset_index(drop=True)

if generar:
    if anio_max < anio_min or calle_max < calle_min or cra_max < cra_min:
        st.error("Revisa los rangos: el máximo no puede ser menor que el mínimo.")
        st.stop()
    params = dict(semilla=semilla, barrios=barrios_txt, vias_rurales=vias_rurales_txt,
                  p_her=p_her, p_dan=p_dan, p_mue=p_mue, anio_min=anio_min, anio_max=anio_max,
                  calle_min=calle_min, calle_max=calle_max, cra_min=cra_min, cra_max=cra_max,
                  n_hot=n_hot, hot_min=hot_min, hot_max=hot_max, n_total=n_total, p_rural=p_rural)
    st.session_state["gen"] = genera(params)
    st.session_state["gen_ciudad"] = ciudad

if "gen" not in st.session_state:
    st.info("Ajusta los parámetros en la barra lateral y pulsa **Generar datos**.")
    st.stop()

df = st.session_state["gen"]
c1, c2, c3 = st.columns(3)
c1.metric("Registros", len(df))
c2.metric("Rurales (excluibles)", int(df["Direccion"].str.contains("KILOMETRO").sum()))
c3.metric("Con muertos", int(df["Accidente con"].str.upper().str.contains("MUERTO").sum()))

cc = st.columns(2)
g = df["Accidente con"].value_counts().reset_index(); g.columns = ["gravedad", "n"]
cc[0].plotly_chart(px.pie(g, names="gravedad", values="n", hole=0.45, title="Gravedad",
                          color_discrete_sequence=px.colors.sequential.YlOrRd_r), use_container_width=True)
cl = df["Clase de Accidente"].str.upper().value_counts().reset_index(); cl.columns = ["clase", "n"]
cc[1].plotly_chart(px.bar(cl, x="n", y="clase", orientation="h", title="Clase",
                          color="n", color_continuous_scale="YlOrRd"), use_container_width=True)

st.dataframe(df.head(20), use_container_width=True)

nombre = "Accidentes_" + "".join(ch for ch in st.session_state["gen_ciudad"].split(",")[0] if ch.isalnum()) + ".csv"
st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                   file_name=nombre, mime="text/csv", type="primary")
