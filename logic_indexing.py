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

def _extrair_textos_csv(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)
    textos = []
    if 'texto' in df.columns:
        textos = df['texto'].dropna().astype(str).tolist()
    return textos

def _extrair_textos_txt(caminho_arquivo):
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        conteudo = f.read()
    return [p.strip() for p in conteudo.split('\n\n') if p.strip()]

def _extrair_textos_pdf(caminho_arquivo):
    textos = []
    try:
        with fitz.open(caminho_arquivo) as doc:
            for pagina in doc:
                texto_pagina = pagina.get_text("text")
                if texto_pagina.strip():
                    paragrafos = [p.strip() for p in texto_pagina.split('\n\n') if p.strip()]
                    textos.extend(paragrafos)
    except Exception as e:
        print(f"\nErro ao processar o PDF {os.path.basename(caminho_arquivo)}: {e}")
    return textos


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