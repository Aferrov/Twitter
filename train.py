import os
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ── Configuración de Hardware y Rutas ──────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"

print(f"==================================================")
print(f"🔥 ENTRENAMIENTO EXCLUSIVO CON HUMAID (Hardware: {device.upper()})")
print(f"==================================================\n")

# ── 1. FUNCIÓN DE LIMPIEZA TEXTUAL ────────────────────────────────
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'http\s+|https\S+', '', texto)
    texto = texto.replace('#', '')
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    texto = re.sub(r'[^\w\s,.:;¡!¿?]', '', texto)
    return " ".join(texto.split())

if __name__ == "__main__":

    # ── 2. CARGA DEL DATASET GLOBAL TRADUCIDO ──────────────────────
    print("[1/5] Cargando archivo 'humaid_es.json'...")
    
    if os.path.exists("humaid_es.json"):
        df_maestro = pd.read_json("humaid_es.json")
        # Cambiamos nombres de columnas a formato estándar (text, label)
        df_maestro = df_maestro.rename(columns={"texto": "text", "etiqueta": "label"})
        print(f"      📥 Base Global (HumAID) cargada con éxito: {len(df_maestro)} tweets.")
        
        # Balanceo/Muestreo inteligente para agilizar el Fine-Tuning en tu GPU
        # Tomamos un máximo de 600 ejemplos por cada categoría (0, 1 y 3)
        df_maestro = df_maestro.groupby("label").sample(n=min(600, len(df_maestro)), random_state=42).reset_index(drop=True)
    else:
        print("      ⚠️ ERROR CRÍTICO: No se encontró 'humaid_es.json'.")
        print("      Por favor, ejecuta primero tu script de descarga y traducción.")
        exit()

    # Procesamiento y limpieza formal para NLP
    df_maestro["text"] = df_maestro["text"].apply(limpiar_texto)
    df_maestro = df_maestro.dropna(subset=["text", "label"])
    df_maestro["label"] = df_maestro["label"].astype(int)
    
    print(f"\n      👉 Volumen final seleccionado para el entrenamiento: {len(df_maestro)} filas.")
    print("Distribución de las clases del dataset global:")
    print(df_maestro["label"].map({0:"0: Sismo", 1:"1: Lluvia", 2:"2: Tráfico", 3:"3: Otros"}).value_counts())

    # ── 3. DIVISIÓN EN ENTRENAMIENTO Y VALIDACIÓN (80% / 20%) ────────
    df_train, df_val = train_test_split(df_maestro, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(df_val.reset_index(drop=True))

    # ── 4. TOKENIZACIÓN CON REGLAS DE BETO ───────────────────────────
    print("\n[2/5] Descargando/Cargando Tokenizer oficial de BETO...")
    tokenizer = AutoTokenizer.from_pretrained(MODELO_BETO)

    def tokenizar_funcion(ejemplos):
        return tokenizer(ejemplos["text"], truncation=True, padding="max_length", max_length=128)

    print("[3/5] Convirtiendo cadenas de texto a tensores estructurados...")
    train_dataset = train_dataset.map(tokenizar_funcion, batched=True)
    val_dataset = val_dataset.map(tokenizar_funcion, batched=True)

    # ── 5. INSTANCIAR ARQUITECTURA DE CLASIFICACIÓN ──────────────────
    print("\n[4/5] Configurando BETO para clasificación lineal (4 neuronas)...")
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_BETO, num_labels=4)

    # Configuración metodológica optimizada para entornos Windows sin redundancia de guardado
    argumentos_entrenamiento = TrainingArguments(
        output_dir="./resultados_entrenamiento",
        eval_strategy="epoch",        # Ejecuta evaluación métrica por cada ciclo terminado
        save_strategy="no",           # Desactiva checkpoints intermedios para cuidar el espacio en disco
        learning_rate=2e-5,           # Hiperparámetro estándar para Fine-Tuning de Transformers
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,           # 3 pasadas completas garantizan convergencia de gradientes
        weight_decay=0.01,            # Penalización L2 para evitar Overfitting
        logging_steps=10,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=argumentos_entrenamiento,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # ── 6. EJECUTAR FINE-TUNING EN GPU (CUDA) ────────────────────────
    print("\n[5/5] 🔥 Iniciando pasadas de entrenamiento y optimización de pesos...")
    trainer.train()
    
    # ── 7. EXPORTACIÓN DEL MODELO RESULTANTE ─────────────────────────
    print("\n💾 Almacenando el cerebro clasificador en './modelo_alertas_arequipa'...")
    model.save_pretrained("./modelo_alertas_arequipa")
    tokenizer.save_pretrained("./modelo_alertas_arequipa")
    
    print("\n==================================================")
    print("🏁 ¡Proceso completado exitosamente!")
    print("==================================================")