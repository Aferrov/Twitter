import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from transformers import AutoTokenizer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
TOPICO_KAFKA = 'alertas-arequipa'

MODELO = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(MODELO)

# Datos de prueba para el flujo
base_datos_tweets = [
    "Urgente: El torrentero de la Av. Venezuela esta por desbordarse por la lluvia. Eviten la zona.",
    "Reportan sismo moderado pero largo aqui en Cayma. Se sintio fuerte.",
    "Tremendo choque pasando el puente Grau en el Cercado. Trafico congestionado."
]

if __name__ == "__main__":
    print("Modulo de Ingesta Activo.")
    id = 1
    
    try:
        while True:
            texto_tweet = random.choice(base_datos_tweets)
            id += 1
            
            tokens = tokenizer(texto_tweet, truncation=True, max_length=128)
            
            datos_pipeline = {
                "id": id,
                "texto": texto_tweet,
                "timestamp": datetime.now().isoformat(),
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            }
            
            producer.send(TOPICO_KAFKA, value=datos_pipeline)
            print(f"Enviado ID: {id} con embeddings a Kafka.")
            
            time.sleep(3.0)
            
    except KeyboardInterrupt:
        print("Ingesta detenida.")
    finally:
        producer.close()