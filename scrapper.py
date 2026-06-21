import os
import re
import json
import time
import pandas as pd
from datetime import datetime
from kafka import KafkaProducer

# =====================================================================
# 1. CONFIGURACIÓN DEL BUS DE EVENTOS (KAFKA)
# =====================================================================
productor_kafka = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topico = 'alertas-arequipa'

# =====================================================================
# 2. DEFINICIÓN DE RUTAS DEL DATA LAKE
# =====================================================================
RUTA_DATA_LAKE = "./data_lake_historico"

# Asegurar que el directorio del Data Lake exista antes de iniciar la ingesta
if not os.path.exists(RUTA_DATA_LAKE):
    os.makedirs(RUTA_DATA_LAKE)
    print(f"Data Lake creado en: '{RUTA_DATA_LAKE}'")

# =====================================================================
# 3. FUNCIÓN DE LIMPIEZA Y NORMALIZACIÓN LÉXICA
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
# 4. PIPELINE DE INGESTA
# =====================================================================
if __name__ == "__main__":
    print("\nIniciando Módulo de Ingesta")
    
    # Archivo de pruebas para ingesta
    archivo_csv = 'dataset_pruebas_expertas.csv'
    
    if not os.path.exists(archivo_csv):
        print(f"No se encontró el archivo '{archivo_csv}' en la ruta actual.")
        exit()
    
    # Se carga el archivo
    try:
        datos_csv = pd.read_csv(archivo_csv)
        datos_csv = datos_csv.fillna("")
        print(f"Dataset cargado exitosamente.")
    except Exception as error:
        print(f"Falló la lectura del archivo CSV: {error}")
        exit()
        
    contador_id = 0
    
    # Interactuamos linea a linea
    try:
        for indice, fila in datos_csv.iterrows():
            contador_id += 1
            
            texto_original = str(fila['text'])
            texto_limpio = limpiar_texto(texto_original)
            
            if not texto_limpio:
                continue
                
            # Estructuramos la alerta para enviar a las capas
            alerta = {
                "id": contador_id,
                "usuario_original": str(fila['user']) if 'user' in fila else "@ciudadano_aqp",
                "texto": texto_limpio,
                "timestamp": datetime.now().isoformat(),
                "estado": str(fila['estado']) if 'estado' in fila else "Informativo",
                "prioridad": str(fila['prioridad']) if 'prioridad' in fila else "Normal"
            }
            
            # CAPA SPEED (ENVIADO VÍA KAFKA)
            productor_kafka.send(topico, value=alerta)
            
            # CAPA BATCH (PERSISTENCIA EN DATA LAKE)
            nombre_archivo_json = f"alerta_{alerta['id']}_{int(time.time())}.json"
            ruta_final_json = os.path.join(RUTA_DATA_LAKE, nombre_archivo_json)
            
            with open(ruta_final_json, 'w', encoding='utf-8') as archivo_json:
                json.dump(alerta, archivo_json, ensure_ascii=False, indent=4)
            
            print(f"ID: {contador_id} -> Enviado | Texto: {texto_limpio[:45]}...")
            
            # Pausa de 2.0 segundos para emular la tasa de llegada en producción
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario.")
    finally:
        productor_kafka.close()
        print("Sistema cerrado correctamente.")