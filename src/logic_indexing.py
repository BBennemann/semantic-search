import os
import logging
from elasticsearch.helpers import bulk, BulkIndexError
from elasticsearch import Elasticsearch, ConnectionError
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from file_processor import (
    extrair_textos_csv,
    extrair_textos_pdf,
    extrair_textos_txt,
    PASTA_DADOS
)

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NOME_DO_INDICE = os.getenv('ELASTICSEARCH_INDEX_NAME', 'buscador_semantico')
MODELO_EMBEDDING = os.getenv('EMBEDDING_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')
ES_HOSTS = os.getenv('ELASTICSEARCH_HOSTS', 'http://localhost:9200').split(',')

DIMENSAO_VETOR = 384
TAMANHO_LOTE_ENCODE = int(os.getenv('ENCODE_BATCH_SIZE', 32))

def carregar_modelo():
    """
    Loads and returns the SentenceTransformer embedding model.
    """
    return SentenceTransformer(MODELO_EMBEDDING)

def conectar_elasticsearch():
    """
    Establishes and returns a connection to the Elasticsearch client.
    """
    try:
        client = Elasticsearch(
            hosts=ES_HOSTS,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30
        )
        if not client.ping():
            logger.error("Falha na conexão inicial com Elasticsearch (ping falhou).")
            raise ConnectionError("Falha na conexão com Elasticsearch.")
        logger.info("Conexão com Elasticsearch estabelecida com sucesso.")
        return client
    except ConnectionError as e:
        logger.error(f"Erro de conexão ao tentar conectar ao Elasticsearch em {ES_HOSTS}: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao conectar ao Elasticsearch: {e}", exc_info=True)
        raise

def criar_indice_se_necessario(client):
    """
    Creates the Elasticsearch index with the correct mapping if it does not already exist.
    """
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
    """
    A generator function that reads files from the data folder, processes them,
    and yields documents ready for Elasticsearch bulk indexing.
    """
    if not os.path.isdir(PASTA_DADOS):
        logger.warning(f"Diretório de dados '{PASTA_DADOS}' não encontrado. Nenhum documento para gerar.")
        return

    logger.info(f"Procurando arquivos em '{PASTA_DADOS}' para gerar documentos.")
    for nome_arquivo in os.listdir(PASTA_DADOS):
        caminho_arquivo = os.path.join(PASTA_DADOS, nome_arquivo)
        textos_fonte = []

        if nome_arquivo.endswith('.csv'):
            textos_fonte = extrair_textos_csv(caminho_arquivo)
        elif nome_arquivo.endswith('.txt'):
            textos_fonte = extrair_textos_txt(caminho_arquivo)
        elif nome_arquivo.endswith('.pdf'):
            textos_fonte = extrair_textos_pdf(caminho_arquivo)
        else:
             logger.debug(f"Ignorando arquivo com extensão não suportada: {nome_arquivo}")
             continue

        if not textos_fonte:
            logger.warning(f"Nenhum texto extraído do arquivo: {nome_arquivo}")
            continue

        logger.info(f"Processando {len(textos_fonte)} chunks do arquivo: {nome_arquivo}")
        for i in range(0, len(textos_fonte), TAMANHO_LOTE_ENCODE):
            lote_textos = textos_fonte[i:i + TAMANHO_LOTE_ENCODE]

            try:
                lote_embeddings = model.encode(lote_textos)
            except Exception as e:
                logger.error(f"Erro ao gerar embeddings para lote do arquivo {nome_arquivo}: {e}", exc_info=True)
                continue

            for texto, embedding in zip(lote_textos, lote_embeddings):
                yield {
                    "_index": NOME_DO_INDICE,
                    "_source": {
                        "texto": texto,
                        "embedding_texto": embedding.tolist(),
                        "fonte_arquivo": nome_arquivo
                    }
                }

def executar_indexacao(client, model):
    """
    Executes the bulk indexing process for all generated documents into Elasticsearch.
    """
    if not os.path.isdir(PASTA_DADOS):
        logger.warning(f"Diretório de dados '{PASTA_DADOS}' não encontrado. Nada a indexar.")
        return 0, 0

    logger.info("Iniciando processo de indexação em lote...")
    documentos_gerados = list(gerar_documentos(model))
    if not documentos_gerados:
        logger.info("Nenhum documento válido encontrado para indexar.")
        return 0, 0

    logger.info(f"Tentando indexar {len(documentos_gerados)} chunks...")
    try:
        sucessos, erros = bulk(
            client=client,
            actions=documentos_gerados,
            raise_on_error=False,
            raise_on_exception=False,
            refresh=True
        )
        if erros:
            logger.error(f"Falha ao indexar {len(erros)} documento(s). Detalhes dos erros: {erros[:5]}")
        logger.info(f"Indexação concluída. Sucessos: {sucessos}, Falhas: {len(erros)}")
        return sucessos, len(erros)
    except BulkIndexError as e:
        logger.error(f"Erro durante a operação de bulk do Elasticsearch: {len(e.errors)} falhas.", exc_info=True)
        logger.error(f"Primeiros erros: {e.errors[:5]}")
        return len(documentos_gerados) - len(e.errors), len(e.errors)
    except Exception as e:
        logger.error(f"Erro crítico inesperado durante a execução do bulk: {e}", exc_info=True)
        return 0, len(documentos_gerados)

def buscar_semantica(client, model, consulta: str, top_k: int = 3):
    """
    Performs a semantic search (KNN) in Elasticsearch based on a user query.
    """
    logger.info(f"Realizando busca semântica para: '{consulta}' com k={top_k}")
    try:
        vetor_consulta = model.encode(consulta)
        query_knn = {
            "field": "embedding_texto",
            "query_vector": vetor_consulta,
            "k": top_k,
            "num_candidates": max(10, top_k * 2)
        }

        response = client.search(
            index=NOME_DO_INDICE,
            knn=query_knn,
            source=["texto", "fonte_arquivo"],
            size=top_k
        )
        logger.info(f"Busca retornou {len(response.get('hits', {}).get('hits', []))} resultados.")
        return response.get('hits', {}).get('hits', [])
    except Exception as e:
        logger.error(f"Erro inesperado durante a busca semântica: {e}", exc_info=True)
        return []

def listar_fontes_indexadas(client):
    """
    Retrieves a list of unique source filenames currently in the Elasticsearch index.
    """
    if not client.indices.exists(index=NOME_DO_INDICE):
        return []

    response = client.search(index=NOME_DO_INDICE, size=0, aggs={"fontes_unicas": {"terms": {"field": "fonte_arquivo", "size":1000}}})
    buckets = response.get('aggregations', {}).get('fontes_unicas', {}).get('buckets', [])
    return [bucket['key'] for bucket in buckets]

def apagar_documentos_por_fonte(client, nome_arquivo: str):
    """
    Deletes all documents from the Elasticsearch index that originated from a specific source file.
    """
    logger.info(f"Tentando apagar documentos da fonte: {nome_arquivo}")
    query = {
        "query": {
            "term": {
                "fonte_arquivo": nome_arquivo
            }
        }
    }
    try:
        response = client.delete_by_query(
            index=NOME_DO_INDICE,
            body=query,
            refresh=True,
            conflicts='proceed'
        )
        apagados = response.get('deleted', 0)
        falhas = response.get('failures', [])
        if falhas:
            logger.error(f"Falhas ao apagar documentos da fonte {nome_arquivo}: {falhas}")
        logger.info(f"Documentos apagados da fonte {nome_arquivo}: {apagados}")
        return apagados
    except Exception as e:
        logger.error(f"Erro inesperado ao apagar documentos da fonte {nome_arquivo}: {e}", exc_info=True)
        raise
