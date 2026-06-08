import os
import sys
import json

# =====================================================================
# 1. CONFIGURACIÓN ABSOLUTA FORZADA PARA EL ENTORNO EN WINDOWS
# =====================================================================
# Forzamos las rutas reales de tu disco C donde se encuentran tus binarios
HADOOP_REAL = r"C:\hadoop"
BIN_REAL = r"C:\hadoop\bin"

os.environ["HADOOP_HOME"] = HADOOP_REAL
os.environ["hadoop.home.dir"] = HADOOP_REAL

# Añadimos el directorio bin al PATH del proceso actual para que Java cargue hadoop.dll
os.environ["PATH"] = BIN_REAL + os.path.pathsep + os.environ.get("PATH", "")
sys.path.append(BIN_REAL)

# Forzar a los Workers distribuidos a usar exactamente tu mismo ejecutable de Python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# OPTIMIZACIÓN EN WINDOWS: Apagar paralelismo redundante de PyTorch en hilos secundarios
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Importaciones oficiales de PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split

# =====================================================================
# 2. INICIALIZACIÓN DEL MODELO INTELIGENTE (Hilo Principal - Main Thread)
# =====================================================================
print("\n📦 Cargando Arquitectura de Deep Learning y pesos de BETO...")
import torch
from transformers import AutoModelForSequenceClassification

MODELO_PATH = "./modelo_alertas_arequipa"
model = AutoModelForSequenceClassification.from_pretrained(MODELO_PATH, num_labels=4)
model.eval()  # Modo evaluación explícito libre de cálculo de gradientes

# Diccionario de mapeo para las predicciones del clasificador
CATEGORIAS = {0: "Sismo", 1: "Lluvia", 2: "Trafico", 3: "Otros"}

# =====================================================================
# 3. PROCESADOR DE MICRO-LOTES (Arquitectura Híbrida de Baja Latencia)
# =====================================================================
def procesar_lote_streaming(df, batch_id):
    # Recolectar el micro-lote de Spark a memoria local de forma ultra-rápida
    alertas_locales = df.collect()
    
    if len(alertas_locales) == 0:
        return

    print(f"\n⚡ [Batch: {batch_id}] - Procesando {len(alertas_locales)} alertas en tiempo real:")
    print("+" + "-"*5 + "+" + "-"*60 + "+" + "-"*12 + "+" + "-"*12 + "+")
    print(f"| {'ID':<3} | {'Texto de la Alerta':<58} | {'Estado':<10} | {'Prioridad':<10} |")
    print("+" + "-"*5 + "+" + "-"*60 + "+" + "-"*12 + "+" + "-"*12 + "+")

    for fila in alertas_locales:
        try:
            # Deserializar el JSON plano que viene del payload string de Kafka
            datos_json = json.loads(fila["value"])
            
            id_alerta = datos_json.get("id", 0)
            texto = datos_json.get("texto", "")
            input_ids = datos_json.get("input_ids", [])
            attention_mask = datos_json.get("attention_mask", [])
            
            if not input_ids or not attention_mask:
                continue

            # Conversión limpia y segura a tensores estables de PyTorch de 64 bits (LongTensor)
            ids_t = torch.tensor([input_ids], dtype=torch.long)
            mask_t = torch.tensor([attention_mask], dtype=torch.long)
            
            # Inferencia directa sobre el hilo actual de Python
            with torch.no_grad():
                outputs = model(input_ids=ids_t, attention_mask=mask_t)
                prediccion = torch.argmax(outputs.logits, dim=1).item()
                
            categoria = CATEGORIAS.get(prediccion, "Otros")
            texto_min = texto.lower()
            
            # --- Reglas de Negocio: Estado Emocional ---
            if any(p in texto_min for p in ["ayuda", "auxilio", "socorro", "atrapados", "heridos", "s.o.s", "sos"]):
                estado = "Ayuda"
            elif any(p in texto_min for p in ["pánico", "panico", "miedo", "terror", "desesperante", "rezos", "dios mío", "dios mio"]):
                estado = "Pánico"
            elif any(p in texto_min for p in ["alcalde", "municipio", "incompetentes", "desgracia", "culpable", "brilla por su ausencia"]):
                estado = "Denuncia"
            else:
                estado = "Informativo"
                
            # --- Reglas de Negocio: Prioridad Operativa ---
            if any(p in texto_min for p in ["urgente", "ahora", "ya", "inmediato", "morir", "ahogar", "colapso", "destruido"]):
                prioridad = "Urgente"
            else:
                prioridad = "Normal"
            
            # Truncar visualmente el texto largo para mantener la estética de la consola
            texto_corto = texto if len(texto) <= 55 else texto[:52] + "..."
            print(f"| {id_alerta:<3} | {texto_corto:<58} | {estado:<10} | {prioridad:<10} |")
            
        except Exception as e:
            print(f"| ERR | Error al procesar alerta individual: {e}")
            
    print("+" + "-"*5 + "+" + "-"*60 + "+" + "-"*12 + "+" + "-"*12 + "+")

# =====================================================================
# 4. ORQUESTADOR DE SPARK STRUCTURED STREAMING
# =====================================================================
if __name__ == "__main__":
    print("\n==================================================")
    print("🚀 CAPA SPEED: Spark Structured Streaming Activa")
    print("==================================================\n")

    # Crear la sesión de Spark adjuntando de manera nativa los conectores de Kafka
    spark = SparkSession.builder \
        .appName("AlertaArequipa-SpeedLayer-Spark") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    # Silenciar logs ruidosos de la JVM de Java
    spark.sparkContext.setLogLevel("WARN")

    # Declarar el flujo de entrada continuo desde Kafka
    df_kafka = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "alertas-arequipa") \
        .option("startingOffsets", "latest") \
        .load()

    # Castear el payload binario ("value") a cadena de texto limpia
    df_string = df_kafka.select(col("value").cast("string"))

    # Despachar las ráfagas continuas de datos mediante el procesador local foreachBatch
    query = df_string.writeStream \
        .foreachBatch(procesar_lote_streaming) \
        .option("checkpointLocation", "./checkpoints_spark_speed") \
        .start()

    query.awaitTermination()