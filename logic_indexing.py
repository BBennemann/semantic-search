import os
import pandas as pd
import fitz  # A biblioteca PyMuPDF é importada como 'fitz'
from elasticsearch.helpers import bulk

# --- Constantes ---
PASTA_DADOS = 'data'
NOME_DO_INDICE = 'buscador_semantico'
DIMENSAO_VETOR = 384

def criar_indice_se_necessario(client):
    """Cria o índice com o mapeamento correto se ele não existir."""
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

def gerar_documentos(model):
    """Lê arquivos .txt, .csv e .pdf da pasta 'data' e gera os documentos para indexação."""
    if not os.path.isdir(PASTA_DADOS):
        return

    for nome_arquivo in os.listdir(PASTA_DADOS):
        caminho_arquivo = os.path.join(PASTA_DADOS, nome_arquivo)

        # Processa arquivos .csv
        if nome_arquivo.endswith('.csv'):
            df = pd.read_csv(caminho_arquivo)
            for _, row in df.iterrows():
                texto = row.get('texto')
                if texto and isinstance(texto, str):
                    yield {
                        "_index": NOME_DO_INDICE,
                        "_source": {
                            "texto": texto,
                            "embedding_texto": model.encode(texto),
                            "fonte_arquivo": nome_arquivo
                        }
                    }
        
        # Processa arquivos .txt
        elif nome_arquivo.endswith('.txt'):
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                # Divide o texto em parágrafos para indexar pedaços menores
                paragrafos = [p.strip() for p in conteudo.split('\n\n') if p.strip()]
                for paragrafo in paragrafos:
                    yield {
                        "_index": NOME_DO_INDICE,
                        "_source": {
                            "texto": paragrafo,
                            "embedding_texto": model.encode(paragrafo),
                            "fonte_arquivo": nome_arquivo
                        }
                    }

        # Processa arquivos .pdf
        elif nome_arquivo.endswith('.pdf'):
            try:
                with fitz.open(caminho_arquivo) as doc:
                    for pagina in doc:
                        texto_pagina = pagina.get_text("text")
                        # Se a página tiver texto, divide em parágrafos
                        if texto_pagina.strip():
                            paragrafos = [p.strip() for p in texto_pagina.split('\n\n') if p.strip()]
                            for paragrafo in paragrafos:
                                yield {
                                    "_index": NOME_DO_INDICE,
                                    "_source": {
                                        "texto": paragrafo,
                                        "embedding_texto": model.encode(paragrafo),
                                        "fonte_arquivo": nome_arquivo
                                    }
                                }
            except Exception as e:
                print(f"\nErro ao processar o PDF {nome_arquivo}: {e}")


def executar_indexacao(client, model):
    """Orquestra a indexação em lote."""
    print("Iniciando indexação em lote...")
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