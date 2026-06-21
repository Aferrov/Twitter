import os
import sys
import json
import torch

# =====================================================================
# 1. CONFIGURACION DEL ENTORNO HADOOP Y SPARK
# =====================================================================
HADOOP_RUTA = r"C:\hadoop"
BIN_RUTA = r"C:\hadoop\bin"

os.environ["HADOOP_HOME"] = HADOOP_RUTA
os.environ["hadoop.home.dir"] = HADOOP_RUTA
os.environ["PATH"] = BIN_RUTA + os.path.pathsep + os.environ.get("PATH", "")
sys.path.append(BIN_RUTA)

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Restringir hilos locales para mitigar la contención de la CPU frente a la Capa Batch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# =====================================================================
# 2. CARGA LOCAL DEL MODELO BETO ABSA CONJUNTO (12 Clases)
# =====================================================================
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Carga del modelo robusto multidimensional utilizado en la tesis
ruta_modelo_absa = "./modelo_beto_absa_arequipa"

print(f"Inicializando el modelo BETO-ABSA desde '{ruta_modelo_absa}'...")
tokenizador = AutoTokenizer.from_pretrained(ruta_modelo_absa)
# CORREGIDO: Se cambia a 12 clases estructurales (4 Aspectos x 3 Sentimientos)
modelo_absa = AutoModelForSequenceClassification.from_pretrained(ruta_modelo_absa, num_labels=12)
modelo_absa.eval()

# Taxonomías analíticas de decodificación de la red neuronal
categorias_desastre = ["Inundación y Lluvias", "Sismos y Derrumbes", "Colapso de Servicios", "Accidentes y Tráfico"]
escalas_severidad = ["Crítico", "En Riesgo", "Estable"]

# =====================================================================
# 3. FUNCION PARA PROCESAR CADA LOTE EN STREAMING (INFERENCIA PURE IA)
# =====================================================================
def procesar_lote(datos_spark, lote_id):
    lista_alertas = datos_spark.collect()
    
    if len(lista_alertas) == 0:
        return

    print(f"\n[SPEED] Lote: {lote_id} - Absorbiendo {len(lista_alertas)} alertas desde Kafka")
    print("---------------------------------------------------------------------------------------")

    alertas_procesadas_batch = []

    for fila in lista_alertas:
        try:
            contenido_json = json.loads(fila["value"])
            alerta_id = contenido_json.get("id", 0)
            texto_alerta = contenido_json.get("texto", "")
            usuario = contenido_json.get("usuario_original", "@anonimo")
            timestamp = contenido_json.get("timestamp", "")
            
            if not texto_alerta:
                continue

            # Tokenización adaptativa de la secuencia textual
            tokens_procesados = tokenizador(
                texto_alerta, 
                truncation=True, 
                padding="max_length", 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Inferencia in-memory protegida contra cálculo de gradientes
            with torch.no_grad():
                resultado = modelo_absa(
                    input_ids=tokens_procesados["input_ids"], 
                    attention_mask=tokens_procesados["attention_mask"]
                )
                prediccion_id = torch.argmax(resultado.logits, dim=1).item()
                
            # DECODIFICACIÓN MATEMÁTICA ABSA CONJUNTA
            # La Inteligencia Artificial ahora asume el control analítico total del flujo
            indice_desastre = prediccion_id // 3   # Dimensión 1: El Aspecto / Tipo de Desastre
            indice_severidad = prediccion_id % 3  # Dimensión 2: El Sentimiento / Severidad Emocional
            
            desastre_ia = categorias_desastre[indice_desastre]
            severidad_ia = escalas_severidad[indice_severidad]
            
            print(f"ID: {alerta_id:<4} | Aspecto: {desastre_ia:<21} | Sentimiento/Severidad: {severidad_ia:<10} | Texto: {texto_alerta[:35]}...")
            
            # Estructurar el documento JSON final para la Serving Layer
            alerta_documento = {
                "id": int(alerta_id),
                "usuario_original": usuario,
                "texto": texto_alerta,
                "timestamp": timestamp,
                "desastre_ia": desastre_ia,
                "severidad_ia": severidad_ia
            }
            alertas_procesadas_batch.append(alerta_documento)
            
        except Exception as error:
            print(f"[ERROR REGISTRO] Falló el cómputo individual: {error}")
            
    print("---------------------------------------------------------------------------------------")
    
    # PERSISTENCIA PARALELA EN LA SERVING LAYER (MONGODB)
    if len(alertas_procesadas_batch) > 0:
        try:
            df_mongo = spark.createDataFrame(alertas_procesadas_batch)
            df_mongo.write \
                .format("mongodb") \
                .mode("append") \
                .option("collection", "alertas_tiempo_real") \
                .save()
            print(f"[SPEED] Sincronizadas {len(alertas_procesadas_batch)} alertas en la colección 'alertas_tiempo_real'.")
        except Exception as e:
            print(f"[CRÍTICO SPEED] Fallo de persistencia streaming en MongoDB: {e}")

# =====================================================================
# 4. CONFIGURACION DE SPARK STRUCTURED STREAMING (MICRO-BATCHES)
# =====================================================================
if __name__ == "__main__":
    print("[SISTEMA] Iniciando Capa Speed con Spark Structured Streaming...")

    mongo_uri_speed = "mongodb://127.0.0.1:27017/tesis_alertas.alertas_tiempo_real"

    # Inicialización del clúster de Spark Streaming aislando hilos de cómputo (local[2])
    spark = SparkSession.builder \
        .appName("CapaSpeedABSA") \
        .master("local[2]") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.write.connection.uri", mongo_uri_speed) \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Suscripción al bus de mensajería inmutable distribuidor
    flujo_kafka = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "alertas-arequipa") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Des-serialización del flujo binario de Kafka a cadenas de caracteres legibles
    flujo_texto = flujo_kafka.select(col("value").cast("string"))

    # Despliegue de la consulta continua por micro-lotes estructurados
    consulta = flujo_texto.writeStream \
        .foreachBatch(procesar_lote) \
        .option("checkpointLocation", "./checkpoints_spark_speed") \
        .start()

    consulta.awaitTermination()