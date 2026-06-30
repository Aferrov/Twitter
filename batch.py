import os
import sys
import json
import re
import torch
import pandas as pd
import glob
from collections import Counter
from datetime import datetime

# =====================================================================
# 1. CONFIGURACIÓN DEL ENTORNO HADOOP Y SPARK
# =====================================================================
HADOOP_RUTA = r"C:\hadoop"
BIN_RUTA = r"C:\hadoop\bin"

os.environ["HADOOP_HOME"] = HADOOP_RUTA
os.environ["hadoop.home.dir"] = HADOOP_RUTA
os.environ["PATH"] = BIN_RUTA + os.path.pathsep + os.environ.get("PATH", "")
sys.path.append(BIN_RUTA)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# =====================================================================
# 2. CARGA LOCAL DEL MODELO BETO DE CLASES DE ABSA (12 CLASES)
# =====================================================================
ruta_modelo_absa = "./modelo_beto_absa_arequipa"

if os.path.exists(ruta_modelo_absa):
    print(f"\nInicializando carga del modelo BETO-ABSA (12 Clases)")
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo_absa)
    modelo_absa = AutoModelForSequenceClassification.from_pretrained(ruta_modelo_absa, num_labels=12)
    modelo_absa.eval()
else:
    print(f"\nNo se encontró el modelo entrenado.")
    exit()

# Listas taxonómicas para la clasificación
categorias_desastre = ["Inundación y Lluvias", "Sismos y Derrumbes", "Colapso de Servicios", "Accidentes y Tráfico"]
escalas_severidad = ["Crítico", "En Riesgo", "Estable"]

# =====================================================================
# 3. CONFIGURACIÓN DE AUDITORÍA Y DETECCIÓN DE DATA DRIFT
# =====================================================================
UMBRAL_DRIFT   = 0.15  # 15% de tolerancia
MIN_FRECUENCIA = 5     # Mínimo de repeticiones para considerar una nueva jerga urbana

STOPWORDS = {
    "de","la","el","en","y","a","los","las","del","se","que","por",
    "con","una","un","es","al","lo","le","su","sus","me","te","nos",
    "pero","como","mas","este","esta","esto","para","fue","han","ha",
    "son","hay","ya","si","no","mi","tu","rt","via","https","http",
}

#Sacar las palabras del vocabulario
def extraer_vocabulario_entrenamiento(ruta_train="split_train.csv"):
    if not os.path.exists(ruta_train):
        print(f"No se encontró '{ruta_train}'. Vocabulario base vacío.")
        return set()
    df = pd.read_csv(ruta_train)
    vocab = set()
    for texto in df["text"].dropna():
        # Normalización para evitar discrepancias por tildes
        texto_norm = texto.lower()
        texto_norm = re.sub(r'[áàäâ]', 'a', texto_norm)
        texto_norm = re.sub(r'[éèëê]', 'e', texto_norm)
        texto_norm = re.sub(r'[íìïî]', 'i', texto_norm)
        texto_norm = re.sub(r'[óòöô]', 'o', texto_norm)
        texto_norm = re.sub(r'[úùüû]', 'u', texto_norm)
        
        tokens = re.findall(r"\b[a-z]{3,}\b", texto_norm)
        vocab.update(t for t in tokens if t not in STOPWORDS)
    print(f"  -> Vocabulario base indexado: {len(vocab)} palabras únicas.")
    return vocab
#Detectar el drift 
def detectar_drift_global(registros_procesados, vocab_entrenamiento):
    """Paso 3: Consolidar estadísticas y aislar registros con deriva semántica."""
    contador_nuevas = Counter()
    tweets_con_drift = []

    for r in registros_procesados:
        if r.get("tiene_drift", False):
            texto_norm = r["texto"].lower()
            texto_norm = re.sub(r'[áàäâ]', 'a', texto_norm)
            texto_norm = re.sub(r'[éèëê]', 'e', texto_norm)
            texto_norm = re.sub(r'[íìïî]', 'i', texto_norm)
            texto_norm = re.sub(r'[óòöô]', 'o', texto_norm)
            texto_norm = re.sub(r'[úùüû]', 'u', texto_norm)
            
            tokens = re.findall(r"\b[a-z]{3,}\b", texto_norm)
            nuevas = [t for t in tokens if t not in vocab_entrenamiento and t not in STOPWORDS]
            
            if nuevas:
                contador_nuevas.update(nuevas)
                tweets_con_drift.append({
                    "id":              r.get("id"),
                    "texto":           r.get("texto"),
                    "palabras_nuevas": ", ".join(nuevas),
                    "desastre":        r.get("desastre"),
                    "severidad":       r.get("severidad"),
                })

    jergas = {p: f for p, f in contador_nuevas.items() if f >= MIN_FRECUENCIA}
    pct = len(tweets_con_drift) / max(len(registros_procesados), 1)
    return pct, jergas, tweets_con_drift

#Guardar el reporte
def guardar_reporte_drift(pct, jergas, tweets_con_drift, total):
    reporte = {
        "timestamp":                  datetime.now().isoformat(),
        "total_tweets":               total,
        "tweets_con_drift":           len(tweets_con_drift),
        "porcentaje_drift":           round(pct, 4),
        "umbral_configurado":         UMBRAL_DRIFT,
        "reentrenamiento_recomendado": pct >= UMBRAL_DRIFT,
        "jergas_detectadas":          dict(
            sorted(jergas.items(), key=lambda x: x[1], reverse=True)[:50]
        ),
    }
    with open("reporte_drift.json", "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    if tweets_con_drift:
        pd.DataFrame(tweets_con_drift).to_csv("tweets_drift_detectado.csv", index=False, encoding="utf-8")
    print(f"Reporte guardado en 'reporte_drift.json'")

# =====================================================================
# 4. PIPELINE PRINCIPAL
# =====================================================================
if __name__ == "__main__":
    print("Extrayendo registros desde el Data Lake.")
    
    patron_archivos = os.path.join("D:\\Twitter\\data_lake_historico", "*.json")
    lista_json = glob.glob(patron_archivos)
    
    alertas_crudas = []
    for ruta_archivo in lista_json:
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                datos_tuit = json.load(f)
                if "texto" in datos_tuit and datos_tuit["texto"].strip():
                    alertas_crudas.append({
                        "id": int(datos_tuit.get("id", 0)),
                        "usuario_original": str(datos_tuit.get("usuario_original", "@anonimo")),
                        "texto": str(datos_tuit["texto"]),
                        "timestamp": str(datos_tuit.get("timestamp", "")),
                        "estado": str(datos_tuit.get("estado", "Informativo")),
                        "prioridad": str(datos_tuit.get("prioridad", "Normal"))
                    })
        except Exception:
            continue

    total_registros = len(alertas_crudas)
    if total_registros == 0:
        print("\nEl Data Lake está vacío. Esperando registros.")
        exit()
        
    print(f"Data Lake cargado. Procesando {total_registros} registros")

    # Inicializar vocabulario base de entrenamiento
    vocab_entrenamiento = extraer_vocabulario_entrenamiento("split_train.csv")
    registros_procesados = []
    
    print("\nIniciando inferencia(12 Clases) y auditoría léxica")
    with torch.no_grad():
        for i, tuit in enumerate(alertas_crudas):
            texto = tuit["texto"]
            
            # Inferencia controlada por tuit
            tokens = tokenizer(texto, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            outputs = modelo_absa(**tokens)
            prediccion_id = torch.argmax(outputs.logits, dim=1).item()
            
            # Decodificación matemática de tu matriz analítica de 12 clases
            indice_desastre = prediccion_id // 3
            indice_severidad = prediccion_id % 3
            
            tuit["desastre"] = categorias_desastre[indice_desastre]
            tuit["severidad"] = escalas_severidad[indice_severidad]
            
            #Evaluar la presencia de Drift léxico
            texto_normalizado = texto.lower()
            texto_normalizado = re.sub(r'[áàäâ]', 'a', texto_normalizado)
            texto_normalizado = re.sub(r'[éèëê]', 'e', texto_normalizado)
            texto_normalizado = re.sub(r'[íìïî]', 'i', texto_normalizado)
            texto_normalizado = re.sub(r'[óòöô]', 'o', texto_normalizado)
            texto_normalizado = re.sub(r'[úùüû]', 'u', texto_normalizado)
            
            tokens_limpios = re.findall(r"\b[a-z]{3,}\b", texto_normalizado)
            tuit["tiene_drift"] = any(
                t not in vocab_entrenamiento and t not in STOPWORDS for t in tokens_limpios
            )
            
            registros_procesados.append(tuit)
            
            if (i + 1) % 100 == 0 or (i + 1) == total_registros:
                print(f" -> Procesados: {i + 1}/{total_registros} tuits históricos.")

    # Calcular el porcentaje global de drift
    print("\nEvaluando métricas de degradación temporal del modelo")
    pct_drift, jergas, tweets_con_drift = detectar_drift_global(registros_procesados, vocab_entrenamiento)
    
    print(f"Tuits con drift: {len(tweets_con_drift)}/{total_registros} ({100*pct_drift:.2f}%)")
    guardar_reporte_drift(pct_drift, jergas, tweets_con_drift, total_registros)

    if pct_drift >= UMBRAL_DRIFT:
        print(f"Drift alto ({100*pct_drift:.1f}%). Se sugiere reentrenamiento.")
    else:
        print(f"Drift bajo ({100*pct_drift:.1f}%). Modelo estable.")

    # =====================================================================
    # 5. MOTOR EN MEMORIA DE SPARK PARA PERSISTENCIA EN MONGODB
    # =====================================================================
    print("\nInicializando motor Apache Spark para persistencia masiva")
    mongo_uri_batch = "mongodb://127.0.0.1:27017/tesis_alertas.alertas_batch"
    
    spark = SparkSession.builder \
        .appName("CapaBatchABSA") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.write.connection.uri", mongo_uri_batch) \
        .config("spark.driver.memory", "4g")\
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    # Convertir la lista en un DataFrame de Spark
    df_final = spark.createDataFrame(registros_procesados)
    
    # Ordenar incluyendo el drift
    df_final = df_final.select(
        col("id").cast("int"),
        col("usuario_original"),
        col("texto"),
        col("timestamp"),
        col("estado"),
        col("prioridad"),
        col("desastre"),
        col("severidad"),
        col("tiene_drift")
    )

    # Persistencia de las alertas en la bd
    print("Persistiendo la base de conocimientos histórica en colección 'alertas_batch' de MongoDB")
    df_final.write \
        .format("mongodb") \
        .mode("append") \
        .option("collection", "alertas_batch") \
        .save()
        
    print("Datos históricos sincronizados con éxito.\n")