import os
import re
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# =====================================================================
# 1. CONFIGURACIÓN DE RUTAS Y HARDWARE
# =====================================================================
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
modelo_base = "dccuchile/bert-base-spanish-wwm-cased"
ruta_salida_absa = "./modelo_beto_absa_arequipa"

print(f"Iniciando Pipeline de Reestructuración y Entrenamiento ABSA en: {dispositivo.upper()}")

# =====================================================================
# 2. FUNCIÓN DE LIMPIEZA DE TEXTO ESTÁNDAR
# =====================================================================
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

# =====================================================================
# 3. LÓGICA DE REMAPEO LÉXICO-CONTEXTUAL PARA EL ABSA DE LA TESIS
# =====================================================================
def transformar_a_clase_absa(fila):
    """
    Toma el desastre original (0 a 3) y evalúa los tokens contextuales del texto
    para inyectar la dimensión de severidad, mapeando al espacio final de 12 clases.
    """
    texto = str(fila['text']).lower()
    desastre_orig = int(fila['label']) # 0, 1, 2 o 3 según HumAID
    
    # Asegurar que el desastre original esté en el rango esperado de 4 categorías macro
    desastre_base = desastre_orig % 4
    
    # Heurística contextual para extraer la severidad intrínseca del tuit real
    # Palabras clave de alta afectación operativa (Severidad: Crítico -> Multiplicador 0)
    if any(w in texto for w in ["urgente", "colapso", "muertos", "heridos", "destruido", "auxilio", "fatal", "desborde"]):
        severidad = 0 
    # Palabras clave de alerta preventiva (Severidad: En Riesgo -> Multiplicador 1)
    elif any(w in texto for w in ["precaucion", "alerta", "riesgo", "lluvia fuerte", "pesado", "trafico", "fuerte"]):
        severidad = 1
    # Por defecto, reportes informativos o de control (Severidad: Estable -> Multiplicador 2)
    else:
        severidad = 2
        
    # FÓRMULA DE ALINEACIÓN DE ETIQUETAS: DesastreBase * 3 + Severidad
    clase_unificada = (desastre_base * 3) + severidad
    return clase_unificada

# =====================================================================
# 4. CARGA, PARTICIÓN TRIPLE Y PREPARACIÓN DEL DATASET
# =====================================================================
if __name__ == "__main__":
    archivo_json = "humaid_es.json"
    
    if not os.path.exists(archivo_json):
        print(f"Error: No se encontró el archivo '{archivo_json}'.")
        exit()
        
    # Cargar y normalizar nombres de columnas originales
    df_json = pd.read_json(archivo_json)
    df_dataframe = df_json.rename(columns={"texto": "text", "etiqueta": "label"})
    
    # Aplicar limpieza de texto estándar
    df_dataframe["text"] = df_dataframe["text"].apply(limpiar_texto)
    df_dataframe = df_dataframe.dropna(subset=["text", "label"])
    
    # Ejecutar la transformación al Espacio Unificado de 12 Clases para el segundo BETO
    print("[DATASET] Aplicando mapeo dimensional a 12 clases combinadas de ABSA...")
    df_dataframe["label"] = df_dataframe.apply(transformar_a_clase_absa, axis=1)
    
    # Limitar el tamaño de muestras por clase combinada para balancear y acelerar el cómputo
    df_dataframe = df_dataframe.groupby("label").sample(
        n=min(250, len(df_dataframe)), random_state=42, replace=True
    ).reset_index(drop=True)
    
    print(f"Total de registros estructurados para el entrenamiento ABSA: {len(df_dataframe)}")
    print(df_dataframe["label"].value_counts().sort_index())

    # PARTICIÓN TRIPLE ESTRATIFICADA (70% Train / 15% Validation / 15% Test)
    df_train, df_temporal = train_test_split(
        df_dataframe, test_size=0.30, random_state=42, stratify=df_dataframe["label"]
    )
    df_val, df_test = train_test_split(
        df_temporal, test_size=0.50, random_state=42, stratify=df_temporal["label"]
    )
    
    # Conversión al formato Dataset nativo de Hugging Face
    dataset_train = Dataset.from_pandas(df_train.reset_index(drop=True))
    dataset_val = Dataset.from_pandas(df_val.reset_index(drop=True))
    dataset_test = Dataset.from_pandas(df_test.reset_index(drop=True))

    # =====================================================================
    # 5. TOKENIZACIÓN VECTORIAL
    # =====================================================================
    tokenizador = AutoTokenizer.from_pretrained(modelo_base)

    def aplicar_tokenizacion(ejemplos):
        return tokenizador(ejemplos["text"], truncation=True, padding="max_length", max_length=128)

    dataset_train = dataset_train.map(aplicar_tokenizacion, batched=True)
    dataset_val = dataset_val.map(aplicar_tokenizacion, batched=True)
    dataset_test = dataset_test.map(aplicar_tokenizacion, batched=True)

    # =====================================================================
    # 6. CONFIGURACIÓN DE LA ARQUITECTURA TRANSFORMER DE 12 NEURONAS
    # =====================================================================
    print("Configurando arquitectura de BETO con 12 salidas lineales para ABSA...")
    modelo = AutoModelForSequenceClassification.from_pretrained(modelo_base, num_labels=12)

    config_entrenamiento = TrainingArguments(
        output_dir="./resultados_checkpoints_absa",
        eval_strategy="epoch",        # Evaluar al finalizar cada ciclo completo
        save_strategy="epoch",
        learning_rate=2e-5,           
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,           # 4 épocas para asegurar la convergencia de la granularidad
        weight_decay=0.01,            
        load_best_model_at_end=True,  # Retener los pesos con menor pérdida de validación
        logging_steps=10,
        fp16=torch.cuda.is_available() # Aceleración CUDA si está habilitada
    )

    entrenador = Trainer(
        model=modelo,
        args=config_entrenamiento,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )

    # =====================================================================
    # 7. FINE-TUNING Y EVALUACIÓN FINAL SOBRE EL CONJUNTO TEST Aislado
    # =====================================================================
    print("[ENTRENAMIENTO] Iniciando ajuste fino sobre las capas de atención...")
    entrenador.train()
    
    print(f"[ALMACENAMIENTO] Guardando el modelo final de la capa Batch en: '{ruta_salida_absa}'")
    modelo.save_pretrained(ruta_salida_absa)
    tokenizador.save_pretrained(ruta_salida_absa)
    
    print("\n[EVALUACIÓN] Ejecutando métricas sobre el conjunto de TEST independiente...")
    metricas_test = entrenador.evaluate(eval_dataset=dataset_test)
    print(f"Métricas finales del modelo en TEST:\n{metricas_test}")
    
    print("Proceso de entrenamiento unificado completado con éxito.")