import json
import time
import re
import os
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from transformers import AutoTokenizer

# 1. CONFIGURACIÓN DEL PRODUCER NATIVO
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
TOPICO_KAFKA = 'alertas-arequipa'

# 2. CARGA DEL TOKENIZADOR DE BETO
MODELO = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(MODELO)

# 3. FUNCIÓN DE LIMPIEZA HOMOGÉNEA (Igual a la de train.py)
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower() # Pasa a minúsculas para estandarizar tokens
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'http\s+|https\S+', '', texto)
    texto = texto.replace('#', '')
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    texto = re.sub(r'[^\w\s,.:;¡!¿?]', '', texto)
    return " ".join(texto.split())

if __name__ == "__main__":
    print("==================================================")
    print("📥 Módulo de Ingesta Activo - Streaming desde CSV")
    print("==================================================\n")
    
    # Nombre del archivo real que contiene tus tweets de prueba
    RUTA_CSV = 'dataset_pruebas_expertas.csv'
    
    if not os.path.exists(RUTA_CSV):
        print(f"❌ ERROR CRÍTICO: No se encuentra el archivo '{RUTA_CSV}' en {os.getcwd()}")
        exit()
        
    try:
        df = pd.read_csv(RUTA_CSV)
        df = df.fillna("")
        print(f"✅ Dataset cargado con éxito. Total de filas a procesar: {len(df)}")
    except Exception as e:
        print(f"❌ Error al leer el archivo CSV: {e}")
        exit()
        
    id_secuencial = 0
    
    try:
        # CORRECCIÓN: Todo este bucle ahora está correctamente indentado dentro del main
        for index, fila in df.iterrows():
            id_secuencial += 1
            
            # Extraer textos del CSV mapeando de forma segura las columnas
            texto_original = str(fila['text'])
            texto_limpio = limpiar_texto(texto_original)
            
            # Si el tuit quedó vacío tras remover enlaces o arrobas, nos lo saltamos
            if not texto_limpio:
                continue
                
            # Tokenizar usando las reglas exactas de BETO (Truncado a 128)
            tokens = tokenizer(texto_limpio, truncation=True, max_length=128)
            
            # Construir el diccionario final para el Pipeline
            datos_pipeline = {
                "id": id_secuencial,
                "usuario_original": str(fila['user']),
                "texto": texto_limpio,
                "timestamp": datetime.now().isoformat(),
                "emocion_csv": str(fila['emotion']),
                "sentimiento_csv": str(fila['sentiment']),
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            }
            
            # Inyectar el payload convertido a bytes en el clúster de Kafka
            producer.send(TOPICO_KAFKA, value=datos_pipeline)
            print(f"📥 [ID: {id_secuencial}] Enviado a Kafka -> '{texto_limpio[:50]}...'")
            
            # Retardo de 2 segundos para simular la llegada en tiempo real en tu dashboard/consola
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\n🛑 Ingesta por CSV detenida manualmente por el usuario.")
    finally:
        print("🔒 Cerrando conexión con el bróker de Kafka...")
        producer.close()