import json
import time
import re
import os
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer
from transformers import AutoTokenizer

# =====================================================================
# 1. CONFIGURACION DE KAFKA
# =====================================================================
# Inicializar el productor de Kafka configurando la serializacion en JSON
productor_kafka = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topico = 'alertas-arequipa'

# =====================================================================
# 2. CARGA DEL TOKENIZADOR
# =====================================================================
# Cargar el tokenizador de BERT para procesar los textos antes de enviarlos
modelo_nombre = "dccuchile/bert-base-spanish-wwm-cased"
tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)

# =====================================================================
# 3. FUNCION DE LIMPIEZA DE TEXTO
# =====================================================================
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    # Pasar a minusculas y remover menciones, enlaces y el caracter hashtag
    texto = texto.lower()
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'http\s+|https\S+', '', texto)
    texto = texto.replace('#', '')
    
    # Reemplazar tildes y caracteres especiales
    texto = texto.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    texto = re.sub(r'[^\w\s,.:;¡!¿?]', '', texto)
    
    # Remover espacios en blanco duplicados
    return " ".join(texto.split())

# =====================================================================
# 4. EJECUCION PRINCIPAL DEL STREAMING DESDE CSV
# =====================================================================
if __name__ == "__main__":
    print("Iniciando modulo de ingesta desde archivo CSV")
    
    archivo_csv = 'dataset_pruebas_expertas.csv'
    
    # Validar la existencia del archivo de datos en el directorio actual
    if not os.path.exists(archivo_csv):
        print(f"Error: No se encontro el archivo '{archivo_csv}' en la ruta actual.")
        exit()
        
    try:
        # Leer el archivo de datos y rellenar celdas vacias
        datos_csv = pd.read_csv(archivo_csv)
        datos_csv = datos_csv.fillna("")
        print(f"Archivo cargado correctamente. Filas totales a procesar: {len(datos_csv)}")
    except Exception as error:
        print(f"Error al leer el archivo de datos: {error}")
        exit()
        
    contador_id = 0
    
    try:
        # Recorrer cada fila del archivo de forma secuencial
        for indice, fila in datos_csv.iterrows():
            contador_id += 1
            
            texto_original = str(fila['text'])
            texto_limpio = limpiar_texto(texto_original)
            
            # Omitir el registro si el texto queda vacio tras la limpieza
            if not texto_limpio:
                continue
                
            # Convertir el texto limpio en secuencias de numeros con el tokenizador
            tokens_procesados = tokenizador(texto_limpio, truncation=True, max_length=128)
            
            # Estructurar el diccionario con el nuevo formato de estado y prioridad
            pay_load = {
                "id": contador_id,
                "usuario_original": str(fila['user']),
                "texto": texto_limpio,
                "timestamp": datetime.now().isoformat(),
                "estado_csv": str(fila['estado']),
                "prioridad_csv": str(fila['prioridad']),
                "input_ids": tokens_procesados["input_ids"],
                "attention_mask": tokens_procesados["attention_mask"]
            }
            
            # Enviar el registro empaquetado al servidor de Kafka
            productor_kafka.send(topico, value=pay_load)
            print(f"Enviado a Kafka - ID: {contador_id} | Texto: {texto_limpio[:45]}...")
            
            # Pausa de 2 segundos para simular la transmision en tiempo real
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nProceso de ingesta detenido por el usuario.")
    finally:
        print("Cerrando la conexion con el servidor de Kafka.")
        productor_kafka.close()