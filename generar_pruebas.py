import os
import pandas as pd

# ── Configuración ──────────────────────────────────────────────────
ARCHIVO_IN  = "humaid_es.json"
ARCHIVO_OUT = "dataset_pruebas_expertas.csv"

print("==================================================")
print("📦 COMPILADOR MASIVO LOCAL (100% OFFLINE)")
print("==================================================\n")

if __name__ == "__main__":
    
    print(f"[1/2] Buscando archivo local de tweets reales '{ARCHIVO_IN}'...")
    
    if not os.path.exists(ARCHIVO_IN):
        print(f"❌ ERROR CRÍTICO: No se encontró el archivo '{ARCHIVO_IN}' en tu directorio.")
        print("Por favor, asegúrate de haber corrido completamente el pipeline de traducción primero.")
        exit()
        
    try:
        # Leer el dataset de tweets reales pre-procesados en español que ya posees
        df_json = pd.read_json(ARCHIVO_IN)
        print(f"      ✅ Archivo origen cargado: {len(df_json)} tweets reales detectados.")
        
        # Mapeamos las columnas al formato exacto que exige tu script ingesta.py
        df_limpio = pd.DataFrame({
            "user": "@humaid_real_user",
            "text": df_json["texto"],  # Tu script de traducción guardó la columna como 'texto'
            "date": "2026-Crisis",
            "emotion": "fear",
            "sentiment": "scared"
        })
        
        # Limpieza sanitaria local para asegurar que no se envíe basura a Kafka
        df_limpio = df_limpio.dropna(subset=["text"])
        df_limpio = df_limpio[df_limpio["text"].astype(str).str.strip() != ""]
        df_limpio = df_limpio.drop_duplicates(subset=["text"])
        
        # Seleccionamos una porción masiva de 3,000 tweets reales para tu validación cruzada
        total_filas = min(3000, len(df_limpio))
        df_final = df_limpio.sample(n=total_filas, random_state=42).reset_index(drop=True)
        
        # Reordenar estructuralmente para el bus de Kafka
        df_final = df_final[["user", "text", "date", "emotion", "sentiment"]]
        
        # Guardar en el disco
        df_final.to_csv(ARCHIVO_OUT, index=False, encoding="utf-8")
        
        print(f"==================================================")
        print(f"🚀 ¡PROCESO COMPLETADO LOCALMENTE!")
        print(f"   Archivo masivo creado: '{ARCHIVO_OUT}'")
        print(f"   Total neto listo para inyectar en Kafka: {len(df_final)} tweets reales.")
        print(f"==================================================")
        
    except Exception as e:
        print(f"❌ ERROR AL CONSOLIDAR EL DATASET LOCAL: {e}")