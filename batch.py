import os
import sys
import json
import torch
import pandas as pd
import glob

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
# 2. CARGA LOCAL DEL MODELO BETO DE CLASES DE ABSA
# =====================================================================
ruta_modelo_absa = "./modelo_beto_absa_arequipa"

if os.path.exists(ruta_modelo_absa):
    print(f"\nInicializando carga del modelo BETO-ABSA")
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
# 3. PIPELINE
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

    # Ejecutamos la inferencia del modelo BETO
    registros_procesados = []
    
    with torch.no_grad():
        for i, tuit in enumerate(alertas_crudas):
            texto = tuit["texto"]
            
            # Inferencia controlada por tuit
            tokens = tokenizer(texto, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            outputs = modelo_absa(**tokens)
            prediccion_id = torch.argmax(outputs.logits, dim=1).item()
            
            # Decodificación matemática de la predicción del modelo
            indice_desastre = prediccion_id // 3
            indice_severidad = prediccion_id % 3
            
            # Asignar las dimensiones analíticas calculadas por el modelo BETO
            tuit["desastre"] = categorias_desastre[indice_desastre]
            tuit["severidad"] = escalas_severidad[indice_severidad]
            
            registros_procesados.append(tuit)
            
            # Imprimir progreso cada 100 registros
            if (i + 1) % 100 == 0 or (i + 1) == total_registros:
                print(f" -> Procesados: {i + 1}/{total_registros} tuits históricos.")

    # =====================================================================
    # 4. MOTOR EN MEMORIA DE SPARK PARA PERSISTENCIA EN MONGODB
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

    # Convertimos la lista en un DataFrame de Spark
    df_final = spark.createDataFrame(registros_procesados)
    
    # Seleccionar y ordenar las columnas
    df_final = df_final.select(
        col("id").cast("int"),
        col("usuario_original"),
        col("texto"),
        col("timestamp"),
        col("estado"),
        col("prioridad"),
        col("desastre"),
        col("severidad")
    )

    #Persistencias de las alertas
    print("Persistiendo la base de conocimientos histórica en colección 'alertas_batch' de MongoDB")
    df_final.write \
        .format("mongodb") \
        .mode("append") \
        .option("collection", "alertas_batch") \
        .save()
        
    print("Datos históricos sincronizados con éxito.\n")