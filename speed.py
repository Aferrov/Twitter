import os
import sys
import json

# =====================================================================
# 1. CONFIGURACION DEL ENTORNO EN WINDOWS
# =====================================================================
# Definir las rutas locales de la carpeta hadoop para evitar errores de Java
HADOOP_RUTA = r"C:\hadoop"
BIN_RUTA = r"C:\hadoop\bin"

os.environ["HADOOP_HOME"] = HADOOP_RUTA
os.environ["hadoop.home.dir"] = HADOOP_RUTA

# Agregar los binarios al path del sistema para cargar hadoop.dll
os.environ["PATH"] = BIN_RUTA + os.path.pathsep + os.environ.get("PATH", "")
sys.path.append(BIN_RUTA)

# Asegurar que Spark use el mismo ejecutable de Python
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Desactivar el uso de multiples hilos en PyTorch para evitar conflictos
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Importar las librerias de PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# =====================================================================
# 2. CARGA DEL MODELO DE INTELIGENCIA ARTIFICIAL
# =====================================================================
print("Cargando modelo y librerias...")
import torch
from transformers import AutoModelForSequenceClassification

# Cargar los pesos del modelo local
ruta_modelo = "./modelo_alertas_arequipa"
modelo = AutoModelForSequenceClassification.from_pretrained(ruta_modelo, num_labels=4)
modelo.eval()

# Diccionario para convertir el numero de la prediccion en texto
categorias_mapa = {0: "Sismo", 1: "Lluvia", 2: "Trafico", 3: "Otros"}

# =====================================================================
# 3. FUNCION PARA PROCESAR CADA LOTE DE DATOS
# =====================================================================
def procesar_lote(datos_spark, lote_id):
    # Pasar los datos del lote actual a una lista local de Python
    lista_alertas = datos_spark.collect()
    
    if len(lista_alertas) == 0:
        return

    print(f"Lote: {lote_id} - Procesando {len(lista_alertas)} alertas")
    print("---------------------------------------------------------------------------------------")

    for fila in lista_alertas:
        try:
            # Convertir el texto de la fila en un diccionario JSON
            contenido_json = json.loads(fila["value"])
            
            alerta_id = contenido_json.get("id", 0)
            texto_alerta = contenido_json.get("texto", "")
            tokens_ids = contenido_json.get("input_ids", [])
            mascara_atencion = contenido_json.get("attention_mask", [])
            
            if not tokens_ids or not mascara_atencion:
                continue

            # Convertir las listas en tensores para el modelo
            tensores_ids = torch.tensor([tokens_ids], dtype=torch.long)
            tensores_mascara = torch.tensor([mascara_atencion], dtype=torch.long)
            
            # Realizar la prediccion con el modelo
            with torch.no_grad():
                resultado = modelo(input_ids=tensores_ids, attention_mask=tensores_mascara)
                prediccion_id = torch.argmax(resultado.logits, dim=1).item()
                
            categoria = categorias_mapa.get(prediccion_id, "Otros")
            texto_minusculas = texto_alerta.lower()
            
            # Determinar el estado segun palabras clave
            if any(palabra in texto_minusculas for palabra in ["ayuda", "auxilio", "socorro", "atrapados", "heridos", "s.o.s", "sos"]):
                estado = "Ayuda"
            elif any(palabra in texto_minusculas for palabra in ["pánico", "panico", "miedo", "terror", "desesperante", "rezos", "dios mío", "dios mio"]):
                estado = "Pánico"
            elif any(palabra in texto_minusculas for palabra in ["alcalde", "municipio", "incompetentes", "desgracia", "culpable", "brilla por su ausencia"]):
                estado = "Denuncia"
            else:
                estado = "Informativo"
                
            # Determinar la prioridad segun palabras clave
            if any(palabra in texto_minusculas for palabra in ["urgente", "ahora", "ya", "inmediato", "morir", "ahogar", "colapso", "destruido"]):
                prioridad = "Urgente"
            else:
                prioridad = "Normal"
            
            print(f"ID: {alerta_id} | Texto: {texto_alerta[:45]}... | Categoria: {categoria} | Estado: {estado} | Prioridad: {prioridad}")
            
        except Exception as error:
            print(f"Error al procesar registro: {error}")
            
    print("---------------------------------------------------------------------------------------")

# =====================================================================
# 4. CONFIGURACION DE SPARK STRUCTURED STREAMING
# =====================================================================
if __name__ == "__main__":
    print("Iniciando capa speed con Spark Streaming")

    # Crear la sesion de Spark con el paquete de Kafka necesario
    spark = SparkSession.builder \
        .appName("CapaSpeedAlertas") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    # Ocultar los mensajes de log secundarios de Java
    spark.sparkContext.setLogLevel("WARN")

    # Configurar la lectura del flujo de datos desde el servidor de Kafka
    flujo_kafka = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "alertas-arequipa") \
        .option("startingOffsets", "latest") \
        .load()

    # Convertir el valor binario de Kafka en una cadena de texto limpia
    flujo_texto = flujo_kafka.select(col("value").cast("string"))

    # Enviar los datos continuos a la funcion de procesamiento por lotes
    consulta = flujo_texto.writeStream \
        .foreachBatch(procesar_lote) \
        .option("checkpointLocation", "./checkpoints_spark_speed") \
        .start()

    consulta.awaitTermination()