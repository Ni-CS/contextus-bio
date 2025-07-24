# scripts/generate_evaluation_data.py (Versão Final - Multimodal)
import argilla as rg
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import base64
from PIL import Image
from io import BytesIO
import torch

# --- CONFIGURAÇÕES ---
# Vamos avaliar a Abordagem 4, que é a nossa mais avançada
APPROACH_ID = "approach_4_unified_hybrid" 
DB_PATH = f"database/{APPROACH_ID}"
MODEL_ID = "openai/clip-vit-large-patch14" # O modelo usado nesta abordagem
TOP_K_RESULTS = 5
DATASET_NAME = "teste"

TEST_QUERIES = [
    "qual a função do retículo endoplasmático?", "diferenças entre célula animal e vegetal",
    "o que é a teoria da endossimbiose?", "estrutura e função dos lisossomos",
    "processo da mitose", "o que são células tronco?", "descrição do complexo de golgi",
    "metamorfose do girino", "o que é a membrana plasmática?", "como funciona a fagocitose?"
]

# --- FUNÇÃO AUXILIAR PARA CONVERTER IMAGEM PARA BASE64 ---
def image_to_base64_html(image_path):
    """Converte uma imagem local em uma string Base64 para embutir em HTML."""
    try:
        with Image.open(image_path) as img:
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_str}" width="200">'
    except Exception as e:
        print(f"Erro ao converter imagem {image_path}: {e}")
        return "<i>Imagem não encontrada</i>"

# --- INICIALIZAÇÃO DO CLIENTE ARGILLA ---
try:
    client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")
    print("Conectado ao servidor Argilla com sucesso.")
except Exception as e:
    print(f"Não foi possível conectar ao Argilla. O Docker Compose está rodando? Erro: {e}")
    exit()

# --- CARREGAMENTO DOS DOIS MOTORES DE BUSCA ---
print("Carregando motores de busca...")
# Carrega o modelo CLIP, que será usado para vetorizar as buscas
from transformers import CLIPModel, CLIPProcessor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Carrega o índice de textos
text_df = pd.read_csv(os.path.join(DB_PATH, 'text_data.csv'))
text_embeddings = np.load(os.path.join(DB_PATH, 'text_embeddings.npy')).astype('float32')
text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
faiss.normalize_L2(text_embeddings)
text_index.add(text_embeddings)

# Carrega o índice de imagens
image_df = pd.read_csv(os.path.join(DB_PATH, 'image_data.csv'))
image_embeddings = np.load(os.path.join(DB_PATH, 'image_embeddings.npy')).astype('float32')
image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
faiss.normalize_L2(image_embeddings)
image_index.add(image_embeddings)

# --- GERAÇÃO DOS REGISTROS (TEXTOS E IMAGENS) ---
records_to_add = []
print("Gerando pares de (busca, resultado) para avaliação...")
for query in TEST_QUERIES:
    # Vetoriza a busca com o CLIP
    with torch.no_grad():
        inputs = processor(text=query, return_tensors="pt").to(device)
        query_vector = model.get_text_features(**inputs).cpu().numpy().astype('float32')
    faiss.normalize_L2(query_vector)
    
    # Busca nos textos
    text_distances, text_indices = text_index.search(query_vector, TOP_K_RESULTS)
    for i, idx in enumerate(text_indices[0]):
        result = text_df.iloc[idx]
        records_to_add.append({
            "fields": {"query": query, "type": "Texto", "content": result['text_for_embedding']}
        })

    # Busca nas imagens
    img_distances, img_indices = image_index.search(query_vector, TOP_K_RESULTS)
    for i, idx in enumerate(img_indices[0]):
        result = image_df.iloc[idx]
        image_html = image_to_base64_html(result['content_path'])
        # Combina a imagem (em HTML) com sua descrição para o avaliador ter todo o contexto
        full_content = f"{image_html}<br><br><b>Descrição:</b> {result['text_for_embedding']}"
        records_to_add.append({
            "fields": {"query": query, "type": "Imagem", "content": full_content}
        })

# --- LÓGICA PARA CARREGAR OU CRIAR O DATASET ---
print(f"Verificando se o dataset '{DATASET_NAME}' já existe no Argilla...")
try:
    dataset = client.datasets(DATASET_NAME)
    print("Dataset encontrado.")
except Exception:
    print("Dataset não encontrado. Criando um novo com as configurações definidas.")
    
    settings = rg.Settings(
        fields=[
            rg.TextField(name="query", title="Busca Realizada"),
            rg.TextField(name="type", title="Tipo de Resultado"),
            # Habilita o markdown para renderizar a tag <img>
            rg.TextField(name="content", title="Resultado Encontrado", use_markdown=True) 
        ],
        questions=[
            rg.RatingQuestion(name="relevancia", title="Este resultado é relevante para a busca?", values=[1, 0])
        ],
        guidelines="Avalie a relevância do resultado. Para imagens, avalie tanto a imagem quanto sua descrição."
    )
    
    dataset = rg.Dataset(name=DATASET_NAME, 
                          settings=settings, 
                          workspace="default")
    
    dataset.create()
    dataset.records.log(records_to_add)

# --- ADIÇÃO DOS REGISTROS ---
""" for record in dataset.records:
    record.delete()  # Limpa registros antigos, se houver """
   


print(f"\nOperação concluída com sucesso! {len(records_to_add)} registros foram adicionados ao dataset '{DATASET_NAME}'!")
print("Sua equipe já pode começar a avaliar em http://localhost:6900")