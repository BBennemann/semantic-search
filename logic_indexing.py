import os
import pandas as pd
import fitz  # A biblioteca PyMuPDF é importada como 'fitz'
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
import streamlit as st

# --- Constantes ---
PASTA_DADOS = 'data'
NOME_DO_INDICE = 'buscador_semantico'
DIMENSAO_VETOR = 384
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'
TAMANHO_LOTE_ENCODE = 32
TAMANHO_MAX_CHUNK = 1024
SOBREPOSICAO_CHUNK = 128

def carregar_modelo():
    return SentenceTransformer(MODELO_EMBEDDING)

def conectar_elasticsearch():
    client = Elasticsearch(
        hosts=["http://localhost:9200"],
        verify_certs=False,
        ssl_show_warn=False
    )
    if not client.ping():
        raise ConnectionError("Falha na conexão com Elasticsearch.")
    return client

def criar_indice_se_necessario(client):
    mapeamento = {
        "properties": {
            "texto": {"type": "text"},
            "embedding_texto": {
                "type": "dense_vector",
                "dims": DIMENSAO_VETOR
            },
            "fonte_arquivo": {"type": "keyword"}
        }
    }
    if not client.indices.exists(index=NOME_DO_INDICE):
        client.indices.create(index=NOME_DO_INDICE, mappings=mapeamento)

def _dividir_texto_em_partes(texto_completo: str) -> list[str]:
    if not texto_completo or len(texto_completo) < TAMANHO_MAX_CHUNK:
        return [texto_completo] if texto_completo else []

    partes = []
    inicio = 0
    while inicio < len(texto_completo):
        fim = inicio + TAMANHO_MAX_CHUNK
        # Garante que o corte não exceda o tamanho do texto
        fim = min(fim, len(texto_completo))
        
        # Pega o pedaço inicial
        parte = texto_completo[inicio:fim]
        partes.append(parte.strip())
        
        # Move o início para a próxima posição, considerando a sobreposição
        proximo_inicio = inicio + TAMANHO_MAX_CHUNK - SOBREPOSICAO_CHUNK
        
        # Se o próximo início for o mesmo ou anterior, evita loop infinito
        if proximo_inicio <= inicio:
            break
        inicio = proximo_inicio
        
    return [p for p in partes if p]

def _extrair_textos_csv(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    textos = []
    if 'texto' in df.columns:
        texto_completo = " ".join(df['texto'].dropna().astype(str).tolist())
        textos = _dividir_texto_em_partes(texto_completo)
    return textos

def _extrair_textos_txt(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    return _dividir_texto_em_partes(conteudo)

def _extrair_textos_pdf(caminho_arquivo):
    texto_completo = ""
    try:
        with fitz.open(caminho_arquivo) as doc:
            for pagina in doc:
                texto_completo += pagina.get_text("text") + "\n"
    except Exception as e:
        print(f"\nErro ao processar o PDF {os.path.basename(caminho_arquivo)}: {e}")
    
    return _dividir_texto_em_partes(texto_completo.strip())


def gerar_documentos(model):
    if not os.path.isdir(PASTA_DADOS):
        return

    for nome_arquivo in os.listdir(PASTA_DADOS):
        caminho_arquivo = os.path.join(PASTA_DADOS, nome_arquivo)
        textos_fonte = []

        if nome_arquivo.endswith('.csv'):
            textos_fonte = _extrair_textos_csv(caminho_arquivo)
        elif nome_arquivo.endswith('.txt'):
            textos_fonte = _extrair_textos_txt(caminho_arquivo)
        elif nome_arquivo.endswith('.pdf'):
            textos_fonte = _extrair_textos_pdf(caminho_arquivo)
        
        for i in range(0, len(textos_fonte), TAMANHO_LOTE_ENCODE): 
            lote_textos = textos_fonte[i:i + TAMANHO_LOTE_ENCODE] 
            
            lote_embeddings = model.encode(lote_textos) 
            
            for texto, embedding in zip(lote_textos, lote_embeddings):
                yield {
                    "_index": NOME_DO_INDICE,
                    "_source": {
                        "texto": texto,
                        "embedding_texto": embedding,
                        "fonte_arquivo": nome_arquivo
                    }
                }


def executar_indexacao(client, model):
    try:
        sucessos, erros = bulk(
            client=client,
            actions=gerar_documentos(model),
            raise_on_error=False
        )
        return sucessos, len(erros)
    except Exception as e:
        print(f"Erro crítico durante a execução do bulk: {e}")
        return 0, -1
    
def buscar_semantica(client, model, consulta: str, top_k: int = 3):
    vetor_consulta = model.encode(consulta)
    query_knn = {
        "field": "embedding_texto",
        "query_vector": vetor_consulta,
        "k": top_k,
        "num_candidates": 10
    }

    response = client.search(
        index=NOME_DO_INDICE,
        knn=query_knn,
        source=["texto", "fonte_arquivo"]
    )
    return response['hits']['hits']

def listar_fontes_indexadas(client):
    if not client.indices.exists(index=NOME_DO_INDICE):
        return []
        
    query = {
        "size": 0,
        "aggs": {
            "fontes_unicas": {
                "terms": {"field": "fonte_arquivo", "size": 1000}
            }
        }
    }
    response = client.search(index=NOME_DO_INDICE, body=query)
    buckets = response['aggregations']['fontes_unicas']['buckets']
    return [bucket['key'] for bucket in buckets]

def apagar_documentos_por_fonte(client, nome_arquivo: str):
    query = {
        "query": {
            "term": {
                "fonte_arquivo": nome_arquivo
            }
        }
    }
    try:
        response = client.delete_by_query(index=NOME_DO_INDICE, body=query, refresh=True)
        apagados = response.get('deleted', 0)
        return apagados
    except Exception as e:
        print(f"Erro ao apagar documentos do ficheiro {nome_arquivo}: {e}")
        raise e