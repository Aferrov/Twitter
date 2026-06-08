# descargar_real_español.py
import urllib.request
import csv
import random
random.seed(42)

BASE = "https://raw.githubusercontent.com/cardiffnlp/xlm-t/main/data/sentiment/spanish/"

ARCHIVOS = {
    "train": ("train_text.txt", "train_labels.txt"),
    "val":   ("val_text.txt",   "val_labels.txt"),
    "test":  ("test_text.txt",  "test_labels.txt"),
}

USUARIOS = [
    "carlos_arequipa","maria_misti","pedro_volcan","lucia_sillar",
    "jose_chili","ana_cayma","roberto_yanahuara","sofia_mollendo",
    "miguel_sabandia","elena_socabaya","diego_cerro","rosa_tiabaya",
    "juan_sachaca","carmen_paucarpata","luis_mariano","patricia_flores",
    "andres_quispe","natalia_bustamante","marco_fernandez","diana_mamani",
]

KEYWORDS = {
    "sismo":   ["sismo","terremoto","temblor","réplica","seísmo","epicentro","sacudió","magnitud","tectónico"],
    "lluvia":  ["huayco","aluvión","inundación","desborde","lluvia","aniego","torrencial","crecida","granizada"],
    "trafico": ["accidente","choque","congestión","tráfico","volcado","atropello","desvío","colisión","semáforo"],
}
EMOTION_MAP = {"sismo":"fear","lluvia":"sadness","trafico":"anger","otros":"neutral"}

def clasificar(texto):
    t = texto.lower()
    for clase, palabras in KEYWORDS.items():
        if any(p in t for p in palabras):
            return clase
    return "otros"

def descargar(url):
    print(f"  Descargando {url.split('/')[-1]}...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8").strip().splitlines()

filas = []

for split, (txt_file, lbl_file) in ARCHIVOS.items():
    print(f"\n[{split.upper()}]")
    try:
        textos   = descargar(BASE + txt_file)
        etiquetas = descargar(BASE + lbl_file)

        for texto, etiqueta in zip(textos, etiquetas):
            texto = texto.strip()
            if len(texto) < 5:
                continue

            # Etiquetas reales: 0=negative, 1=neutral, 2=positive
            lbl = int(etiqueta.strip())
            sentiment = {0:"negative", 1:"neutral", 2:"positive"}.get(lbl, "neutral")
            clase     = clasificar(texto)
            emotion   = EMOTION_MAP[clase]

            filas.append({
                "text":      texto,
                "user":      random.choice(USUARIOS),
                "emotion":   emotion,
                "sentiment": sentiment,
            })
        print(f"  ✅ {len(textos)} tweets reales cargados")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# ── Guardar CSV ───────────────────────────────────────────────────
random.shuffle(filas)
ARCHIVO = "sentiment_analysis_dataset.csv"
with open(ARCHIVO, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text","user","emotion","sentiment"])
    writer.writeheader()
    writer.writerows(filas)

from collections import Counter
print(f"\n{'='*55}")
print(f"✅ Guardado: '{ARCHIVO}'  ({len(filas)} tweets REALES en español)")
print(f"\n   Sentiment (etiquetas reales del dataset):")
for k,v in Counter(r["sentiment"] for r in filas).most_common():
    barra = "█" * (v // 10)
    print(f"     {k:<12}: {v:>4}  {barra}")
print(f"\n   Emotion (por keywords):")
for k,v in Counter(r["emotion"] for r in filas).most_common():
    barra = "█" * (v // 10)
    print(f"     {k:<12}: {v:>4}  {barra}")
print(f"\n   3 tweets de muestra:")
for f in filas[:3]:
    print(f"     → {f['text'][:70]}")
    print(f"       sentiment={f['sentiment']}  emotion={f['emotion']}\n")
print(f"Ejecuta tu scrapper:")
print(f"  python scrapper.py")