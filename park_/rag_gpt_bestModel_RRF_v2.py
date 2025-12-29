import os
import json
import traceback
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder

from openai import OpenAI


# =========================
# 0) 환경/모델 설정
# =========================
_ = load_dotenv()

# ✅ 다국어 임베딩 모델 (768-dim)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = int(embed_model.get_sentence_embedding_dimension())  # ✅ dims 자동 감지

# ✅ 다국어 reranker 모델 (CrossEncoder)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# OpenAI 설정
client = OpenAI()
llm_model = "gpt-4o-mini"

# ES 설정
es_username = "elastic"
es_password = "x9lxBdj3-1O-yi8+4sFA"
ES_INDEX = "test"

es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=(es_username, es_password),
    ca_certs="/opt/elasticsearch-8.8.0/config/certs/http_ca.crt"
)

print(es.info())


# =========================
# 1) Chunking
# =========================
def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 150) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if len(chunk) >= 30:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_chunk_docs(docs: List[Dict[str, Any]], chunk_size: int = 700, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    chunked = []
    for d in docs:
        docid = d.get("docid")
        content = d.get("content", "")
        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for ci, c in enumerate(chunks):
            chunked.append({
                "docid": docid,
                "chunk_id": f"{docid}_{ci}",
                "content": c
            })
    return chunked


# =========================
# 2) Multilingual Embedding (E5 포맷)
# =========================
def e5_passage(text: str) -> str:
    return f"passage: {text}"

def e5_query(text: str) -> str:
    return f"query: {text}"

def get_embedding_passages(passages: List[str]) -> List[List[float]]:
    vecs = embed_model.encode(
        [e5_passage(p) for p in passages],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vecs.tolist()

def get_embedding_query(query: str) -> List[float]:
    vec = embed_model.encode(
        [e5_query(query)],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]
    return vec.tolist()

def get_embeddings_in_batches(docs: List[Dict[str, Any]], batch_size: int = 128) -> List[List[float]]:
    all_vecs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        passages = [doc["content"] for doc in batch]
        vecs = get_embedding_passages(passages)
        all_vecs.extend(vecs)
        print(f"embedding batch {i} ~ {i+len(batch)}")
    return all_vecs

# =========================
# 3) ES 인덱스/색인
# =========================
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

mappings = {
    "properties": {
        "docid": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "content": {"type": "text", "analyzer": "nori"},
        "embeddings": {
            "type": "dense_vector",
            "dims": EMBED_DIM,
            "index": True,
            "similarity": "cosine"
        }
    }
}

def create_es_index(index: str, settings: Dict[str, Any], mappings: Dict[str, Any]) -> None:
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)

def bulk_add(index: str, docs: List[Dict[str, Any]]) -> Any:
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions)


# =========================
# 4) 검색 (BM25 + KNN 후보 생성) + ✅ RRF 결합
# =========================
def sparse_retrieve(query_str: str, size: int = 50):
    query = {
        "match": {
            "content": {
                "query": query_str,
                "operator": "and",
                "minimum_should_match": "70%"
            }
        }
    }
    return es.search(index=ES_INDEX, query=query, size=size)

def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 200):
    query_vec = get_embedding_query(query_str)
    knn = {"field": "embeddings", "query_vector": query_vec, "k": size, "num_candidates": num_candidates}
    return es.search(index=ES_INDEX, knn=knn)

def _chunk_id_from_hit(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {})
    return str(src.get("chunk_id", ""))

def rrf_merge_hits(
    bm25_hits: List[Dict[str, Any]],
    knn_hits: List[Dict[str, Any]],
    limit: int = 120,
    k: int = 60,
    w_bm25: float = 1.0,
    w_knn: float = 1.0
) -> List[Dict[str, Any]]:
    """
    ✅ RRF(Reciprocal Rank Fusion)
    score(d) = w_bm25/(k + rank_bm25(d)) + w_knn/(k + rank_knn(d))

    - rank는 1부터 시작
    - k는 보통 60(안정화 상수)
    - w_*는 원하면 가중치 줄 수 있는데, 기본은 동일(1.0)
    """
    scores: Dict[str, float] = {}
    hit_map: Dict[str, Dict[str, Any]] = {}

    # BM25 ranks
    for i, h in enumerate(bm25_hits):
        cid = _chunk_id_from_hit(h)
        if not cid:
            continue
        hit_map[cid] = h
        rank = i + 1
        scores[cid] = scores.get(cid, 0.0) + (w_bm25 / (k + rank))

    # KNN ranks
    for i, h in enumerate(knn_hits):
        cid = _chunk_id_from_hit(h)
        if not cid:
            continue
        hit_map.setdefault(cid, h)
        rank = i + 1
        scores[cid] = scores.get(cid, 0.0) + (w_knn / (k + rank))

    # RRF 점수로 정렬 후 top limit
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_hits = []
    for cid, rrf_score in ranked[:limit]:
        h = hit_map[cid]
        # 디버깅/분석용: RRF 점수 저장(선택)
        h["_rrf_score"] = float(rrf_score)
        merged_hits.append(h)

    return merged_hits


# =========================
# 5) Reranker (CrossEncoder)
# =========================
def rerank(query: str, hits: List[Dict[str, Any]], topn: int = 20) -> List[Tuple[float, Dict[str, Any]]]:
    pairs = [(query, h["_source"]["content"]) for h in hits]
    if not pairs:
        return []
    scores = reranker.predict(pairs)
    scored = list(zip(scores.tolist(), hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]


def select_topk_docids(scored_hits: List[Tuple[float, Dict[str, Any]]], k_doc: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
    best_by_doc = {}
    best_chunk_by_doc = {}

    for score, hit in scored_hits:
        src = hit["_source"]
        docid = src.get("docid")
        if docid is None:
            continue
        if (docid not in best_by_doc) or (score > best_by_doc[docid]):
            best_by_doc[docid] = score
            best_chunk_by_doc[docid] = {
                "score": float(score),
                "content": src.get("content", ""),
                "chunk_id": src.get("chunk_id", "")
            }

    doc_sorted = sorted(best_by_doc.items(), key=lambda x: x[1], reverse=True)
    top_docids = [d for d, _ in doc_sorted[:k_doc]]
    references = [best_chunk_by_doc[d] for d in top_docids]
    return top_docids, references


# =========================
# 6) LLM 프롬프트/도구 정의
# =========================
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


# =========================
# 7) RAG 파이프라인 (Hybrid + ✅RRF + Rerank)
# =========================
def hybrid_search_with_rerank(
    query: str,
    k_final: int = 3,
    bm25_k: int = 50,
    knn_k: int = 50,
    rrf_k: int = 60,
    rrf_limit: int = 120,
    w_bm25: float = 1.0,
    w_knn: float = 1.0
) -> Tuple[List[str], List[Dict[str, Any]]]:
    bm25 = sparse_retrieve(query, size=bm25_k)
    knn = dense_retrieve(query, size=knn_k)

    bm25_hits = bm25["hits"]["hits"]
    knn_hits = knn["hits"]["hits"]

    # ✅ RRF로 후보 결합
    candidates = rrf_merge_hits(
        bm25_hits=bm25_hits,
        knn_hits=knn_hits,
        limit=rrf_limit,
        k=rrf_k,
        w_bm25=w_bm25,
        w_knn=w_knn
    )

    # reranker로 최종 정렬
    reranked = rerank(query, candidates, topn=40)

    topk_docids, references = select_topk_docids(reranked, k_doc=k_final)
    return topk_docids, references


def answer_question(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "search"}},
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception:
        traceback.print_exc()
        return response

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query", "")

        topk_docids, references = hybrid_search_with_rerank(
            standalone_query,
            k_final=3,
            bm25_k=50,
            knn_k=50,
            rrf_k=60,
            rrf_limit=120,
            w_bm25=1.0,
            w_knn=1.0
        )

        response["standalone_query"] = standalone_query
        response["topk"] = topk_docids
        response["references"] = references

        retrieved_context = [r["content"] for r in references]
        content = json.dumps(retrieved_context, ensure_ascii=False)

        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages

        try:
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            )
        except Exception:
            traceback.print_exc()
            return response

        response["answer"] = qaresult.choices[0].message.content
    else:
        response["answer"] = result.choices[0].message.content

    return response


def eval_rag(eval_filename: str, output_filename: str):
    with open(eval_filename, "r", encoding="utf-8") as f, open(output_filename, "w", encoding="utf-8") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f"Test {idx}\nQuestion: {j['msg']}")
            response = answer_question(j["msg"])
            print(f"Answer: {response['answer']}\n")

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(f"{json.dumps(output, ensure_ascii=False)}\n")
            idx += 1


# =========================
# 8) 실행부: (재)색인 + 평가
# =========================
if __name__ == "__main__":
    create_es_index(ES_INDEX, settings, mappings)

    with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    chunked_docs = make_chunk_docs(raw_docs, chunk_size=700, chunk_overlap=150)
    print(f"raw docs: {len(raw_docs)} -> chunked docs: {len(chunked_docs)}")

    embeddings = get_embeddings_in_batches(chunked_docs, batch_size=128)

    index_docs = []
    for doc, emb in zip(chunked_docs, embeddings):
        doc["embeddings"] = emb
        index_docs.append(doc)

    ret = bulk_add(ES_INDEX, index_docs)
    print(ret)

    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    topk, refs = hybrid_search_with_rerank(test_query, k_final=3, bm25_k=20, knn_k=20)
    print("TOPK docids:", topk)
    for r in refs:
        print("rerank_score:", r["score"], "chunk_id:", r["chunk_id"])
        print("content:", r["content"][:200], "...\n")

    eval_rag("../data/eval.jsonl", "sample_submission.csv")