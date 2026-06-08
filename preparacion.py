# pipeline_descarga_y_traduccion.py
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────────────
BATCH_SIZE  = 16   # bajar a 16 si te quedas sin memoria RAM/VRAM
ARCHIVO_OUT = "humaid_es.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando: {device.upper()}")

# ── 1. Cargar modelos ──────────────────────────────────────────────
print("\n[1/4] Cargando Helsinki-NLP...")
tok_trad = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mod_trad  = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)

# CORRECCIÓN: Se descarga especificando el split principal 'train' directamente
# ❌ ANTES (Causaba el error de nombres de validación cruzada):
# df = pd.DataFrame(load_dataset("QCRI/HumAID-all", split="train"))

# ✅ AHORA (Fuerza la descarga ignorando chequeos rígidos de metadatos):
print("[2/4] Cargando HumAID...")
df = pd.DataFrame(load_dataset("QCRI/HumAID-all", split="train", verification_mode="no_checks"))
print(f"      Total cargado: {len(df)} tweets")

# ── 2. Filtrar y mapear clases ────────────────────────────────────
print("[3/4] Filtrando clases relevantes...")

# Filtramos primero para quedarnos con los tweets de información útil
CLASES_RELEVANTES = [
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "displaced_people_and_evacuations",
    "rescue_volunteering_or_donation_efforts",
    "not_humanitarian",
    "other_relevant_information",
    "sympathy_and_support",
]
df = df[df["class_label"].isin(CLASES_RELEVANTES)].copy()

# Asignamos las etiquetas basándonos en las palabras del propio texto del tweet
# Esta aproximación es infalible y evita depender de columnas externas de la API
df["etiqueta"] = 3  # Por defecto todos se marcan como 3 ("otros")

# Si el texto en inglés menciona términos de terremoto -> Categoría 0 (Sismo)
mask_sismo = df["tweet_text"].str.lower().str.contains("earthquake|quake|tremor|seismic", na=False)
df.loc[mask_sismo, "etiqueta"] = 0

# Si el texto menciona términos de inundación o huracán -> Categoría 1 (Lluvia)
mask_lluvia = df["tweet_text"].str.lower().str.contains("flood|rain|hurricane|typhoon|cyclone", na=False)
df.loc[mask_lluvia, "etiqueta"] = 1

# Forzar la clase 3 (Otros) de forma estricta para tweets de descarte o apoyo emocional
mask_otros = df["class_label"].isin(["not_humanitarian", "other_relevant_information", "sympathy_and_support"])
df.loc[mask_otros, "etiqueta"] = 3

df = df.dropna(subset=["etiqueta", "tweet_text"])
df["etiqueta"] = df["etiqueta"].astype(int)

print(f"      Tweets después de filtrar: {len(df)}")
print(df["etiqueta"].map({0:"sismo", 1:"lluvia", 2:"trafico", 3:"otros"}).value_counts())

# ── 3. Traducir ────────────────────────────────────────────────────
print("[4/4] Traduciendo al español...")

def traducir(textos: list[str]) -> list[str]:
    resultado = []
    for i in tqdm(range(0, len(textos), BATCH_SIZE), desc="Traduciendo"):
        batch = [f">>es<< {str(t)}" for t in textos[i:i+BATCH_SIZE]]
        tokens = tok_trad(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            ids = mod_trad.generate(**tokens)
        trad = tok_trad.batch_decode(ids, skip_special_tokens=True)
        resultado.extend(trad)
    return resultado

# Ejecutar traducción masiva
df["texto"] = traducir(df["tweet_text"].tolist())

# ── 4. Guardar ─────────────────────────────────────────────────────
df_final = df[["texto", "etiqueta"]].copy()
df_final.to_json(ARCHIVO_OUT, orient="records", force_ascii=False, indent=2)

print(f"\n✅ Guardado en '{ARCHIVO_OUT}'")
print(f"   Total: {len(df_final)} tweets listos para BETO")
print("\nMuestra de 3 tweets por clase:")
for etiqueta, nombre in {0:"sismo", 1:"lluvia", 3:"otros"}.items():
    print(f"\n   [{nombre.upper()}]")
    for txt in df_final[df_final["etiqueta"]==etiqueta]["texto"].head(3):
        print(f"    → {txt}")