# scripts/generate_evaluation_data.py (Versão Definitiva - Corrigida com base no código-fonte)
import argilla as rg
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# --- CONFIGURAÇÕES ---
APPROACH_ID = "approach_1_text_only" 
DB_PATH = f"database/{APPROACH_ID}"
MODEL_ID = 'intfloat/multilingual-e5-large'
TOP_K_RESULTS = 5
DATASET_NAME = "avaliacao-final-contextus-bio"

TEST_QUERIES = [
    "qual a função do retículo endoplasmático?", "diferenças entre célula animal e vegetal",
    "o que é a teoria da endossimbiose?", "estrutura e função dos lisossomos",
    "processo da mitose", "o que são células tronco?", "descrição do complexo de golgi",
    "metamorfose do girino", "o que é a membrana plasmática?", "como funciona a fagocitose?"
]

# --- INICIALIZAÇÃO DO CLIENTE ARGILLA ---
try:
    client = rg.Argilla(api_url="http://localhost:6900", api_key="argilla.apikey")
    print("Conectado ao servidor Argilla com sucesso.")
except Exception as e:
    print(f"Não foi possível conectar ao Argilla. O Docker Compose está rodando? Erro: {e}")
    exit()

# --- CARREGAMENTO DO MOTOR DE BUSCA ---
print("Carregando motor de busca...")
model = SentenceTransformer(MODEL_ID)
text_df = pd.read_csv(os.path.join(DB_PATH, 'text_data.csv'))
text_embeddings = np.load(os.path.join(DB_PATH, 'text_embeddings.npy')).astype('float32')
text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
faiss.normalize_L2(text_embeddings)
text_index.add(text_embeddings)

# --- GERAÇÃO DOS REGISTROS ---
records_to_add = []
print("Gerando pares de (busca, resultado) para avaliação...")
for query in TEST_QUERIES:
    query_vector = model.encode(["query: " + query]).astype('float32')
    faiss.normalize_L2(query_vector)
    distances, indices = text_index.search(query_vector, TOP_K_RESULTS)
    
    for i, idx in enumerate(indices[0]):
        result = text_df.iloc[idx]
        record_dict = {
            "query": query,
            "result_text": result['text_for_embedding'],
            "page": f"Página {int(result['page'])}",
            "score": f"Score: {distances[0][i]:.4f}"
        }
        records_to_add.append(record_dict)

# --- LÓGICA CORRIGIDA PARA CARREGAR OU CRIAR O DATASET ---
print(f"Verificando se o dataset '{DATASET_NAME}' já existe no Argilla...")
try:
    # A forma correta de buscar um dataset por nome na nova API
    dataset = client.datasets(DATASET_NAME)
    print("Dataset encontrado. Apenas adicionando novos registros.")
except Exception: # Se o 'from_name' falhar (ex: NotFoundError), criamos um novo.
    print("Dataset não encontrado. Criando um novo com as configurações definidas.")
    
    # 1. Definimos a configuração primeiro
    dataset_settings = rg.Settings(
        fields=[
            rg.TextField(name="query"),
            rg.TextField(name="result_text"),
            rg.TextField(name="page"),
            rg.TextField(name="score")
        ],
        questions=[
            rg.RatingQuestion(
                name="relevancia",
                title="Este resultado é relevante para a busca?",
                description="1 = Relevante, 0 = Não Relevante.",
                values=[1, 0] 
            )
        ],
        guidelines="Avalie a relevância do resultado em relação à busca."
    )
    
    # 2. Criamos uma instância local do dataset com as configurações
    dataset = rg.Dataset(
        name=DATASET_NAME,
        settings=dataset_settings,
        workspace="default",  # Use o workspace padrão
    )
    
    # 3. Criamos o dataset no servidor
    dataset.create()

# --- Adicionamos os registros ao dataset (que agora garantidamente existe) ---
""" for record in dataset.records:
    record.delete()  # Limpa registros antigos, se houver """
dataset.records.log(records_to_add)

print(f"\nOperação concluída com sucesso para o dataset '{DATASET_NAME}'!")
print("Sua equipe já pode começar a avaliar em http://localhost:6900 (usuário: argilla, senha: 12345678)")