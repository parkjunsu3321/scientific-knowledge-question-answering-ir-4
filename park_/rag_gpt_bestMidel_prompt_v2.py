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
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

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
    """
    간단한 문자 기반 청킹.
    - chunk_size / chunk_overlap: "문자 기준" (언어 불문 안정적으로 동작)
    - 너무 짧은 chunk는 제거
    """
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
    """
    입력 docs(원문) -> chunk 단위 docs로 변환
    documents.jsonl 각 라인에 최소한 {"docid":..., "content":...} 가 있다고 가정
    """
    chunked = []
    for d in docs:
        docid = d.get("docid")
        content = d.get("content", "")
        chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for ci, c in enumerate(chunks):
            chunked.append({
                "docid": docid,                 # 원문 doc id 유지
                "chunk_id": f"{docid}_{ci}",     # chunk 식별자
                "content": c                    # ES 검색 대상은 chunk content
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
    """
    E5 계열은 query/passage prefix 권장 + cosine 유사도 권장
    normalize_embeddings=True 로 코사인 최적화
    """
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
            "dims": 768,
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
# 4) 검색 (BM25 + KNN 후보 생성)
# =========================
def sparse_retrieve(query_str: str, size: int = 50):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    return es.search(index=ES_INDEX, query=query, size=size, sort="_score")

def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 200):
    query_vec = get_embedding_query(query_str)
    knn = {
        "field": "embeddings",
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=ES_INDEX, knn=knn)

def merge_hits(bm25_hits, knn_hits, limit: int = 100) -> List[Dict[str, Any]]:
    merged = {}
    for h in bm25_hits:
        cid = h["_source"].get("chunk_id")
        if cid and cid not in merged:
            merged[cid] = h
    for h in knn_hits:
        cid = h["_source"].get("chunk_id")
        if cid and cid not in merged:
            merged[cid] = h

    # (초기 후보는 ES score 기반으로 대충 정렬해서 limit로 자름)
    cand = list(merged.values())
    cand.sort(key=lambda x: (x.get("_score", 0.0)), reverse=True)
    return cand[:limit]


# =========================
# 5) Reranker (CrossEncoder)
# =========================
def rerank(query: str, hits: List[Dict[str, Any]], topn: int = 30) -> List[Tuple[float, Dict[str, Any]]]:
    pairs = [(query, h["_source"]["content"]) for h in hits]
    if not pairs:
        return []

    scores = reranker.predict(pairs)  # numpy array
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
## Role: 지식 전문가 (Search-grounded QA)

## Core Principle (가장 중요)
- 모든 답변은 반드시 search API를 통해 얻은 검색 결과에만 기반해야 한다.
- 검색 결과에 포함되지 않은 내용은 절대로 추론하거나 보완해서 사용하지 않는다.
- LLM의 사전 학습 지식, 일반 상식, 암묵적 배경지식 사용을 엄격히 금지한다.

## Instructions
1. 사용자의 질문에 답변하기 전에, 반드시 search API 결과를 확인하고 그 내용만을 사용한다.
2. 검색 결과에서 직접적으로 확인 가능한 정보만 요약·재구성하여 답변한다.
3. 검색 결과에 없는 정보는 다음과 같이 답변한다:
   - "제공된 검색 결과만으로는 해당 질문에 대한 정보를 확인할 수 없습니다."
4. 추측, 일반화, 보완 설명, 추가 배경 설명을 하지 않는다.
5. 수치, 날짜, 고유명사, 원인-결과 관계는 검색 결과에 명시된 경우에만 사용한다.
6. 여러 검색 결과가 있을 경우, 공통적으로 일치하는 정보만 사용한다.
7. 사용자의 이전 대화 맥락은 질문 의도 파악에만 사용하고, 지식 보완에는 사용하지 않는다.
8. 반드시 한국어로 간결하고 명확하게 답변한다.

## Output Style
- 불필요한 서론 없이 핵심 정보 위주로 작성
- 검색 결과 기반 사실 서술 위주
- 의견, 평가, 해석 표현 금지
"""

persona_function_calling = """
## Role: 지식 전문가 (Strict Search-first Agent)

## Core Rule
- 지식, 사실, 개념, 정의, 비교, 현황, 원인, 통계, 기술 설명이 포함된 질문은
  반드시 search API를 호출해야 한다.
- search 없이 직접 답변하는 것을 금지한다.

## Instruction
1. 사용자의 질문이 다음 중 하나라도 포함하면 반드시 search API를 호출한다:
   - 사실 확인 (What / When / Who / Why / How)
   - 개념 설명, 정의, 원리
   - 기술, 모델, 알고리즘, 논문, 제품, 회사, 정책
   - 수치, 통계, 성능 비교, 동향
2. search API 호출 후에만 답변 생성이 가능하다.
3. search 결과가 없거나 불충분한 경우:
   - search 결과가 부족함을 명확히 알리고 답변을 중단한다.
4. search 결과를 통해서만 답변하며, LLM의 내부 지식은 보조적으로도 사용하지 않는다.
5. 지식과 무관한 대화(인사, 감정 표현, 요청 확인 등)는 자연스럽게 응답한다.

## Forbidden
- "일반적으로", "보통", "알려져 있다" 등의 표현 사용 금지
- search 결과에 없는 배경 설명 추가 금지
- LLM 추론 기반 보완 설명 금지
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
# 7) RAG 파이프라인 (Hybrid + Rerank)
# =========================
def hybrid_search_with_rerank(query: str, k_final: int = 10, bm25_k: int = 50, knn_k: int = 50) -> Tuple[List[str], List[Dict[str, Any]]]:
    bm25 = sparse_retrieve(query, size=bm25_k)
    knn = dense_retrieve(query, size=knn_k)

    bm25_hits = bm25["hits"]["hits"]
    knn_hits = knn["hits"]["hits"]

    candidates = merge_hits(bm25_hits, knn_hits, limit=120)
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

        # ✅ Hybrid + Rerank 검색
        topk_docids, references = hybrid_search_with_rerank(standalone_query, k_final=10, bm25_k=50, knn_k=50)

        response["standalone_query"] = standalone_query
        response["topk"] = topk_docids
        response["references"] = references

        # LLM에 넘길 컨텍스트(선택된 references의 content만)
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
    with open(eval_filename) as f, open(output_filename, "w") as of:
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
    # 1) 인덱스 생성
    create_es_index(ES_INDEX, settings, mappings)

    # 2) 원문 로드
    with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    # 3) ✅ 청킹
    chunked_docs = make_chunk_docs(raw_docs, chunk_size=700, chunk_overlap=150)
    print(f"raw docs: {len(raw_docs)} -> chunked docs: {len(chunked_docs)}")

    # 4) ✅ chunk 단위 임베딩 생성
    embeddings = get_embeddings_in_batches(chunked_docs, batch_size=128)

    # 5) ES 색인용 문서 구성
    index_docs = []
    for doc, emb in zip(chunked_docs, embeddings):
        doc["embeddings"] = emb
        index_docs.append(doc)

    # 6) bulk indexing
    ret = bulk_add(ES_INDEX, index_docs)
    print(ret)

    # (선택) 간단 검색 테스트
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    topk, refs = hybrid_search_with_rerank(test_query, k_final=10, bm25_k=20, knn_k=20)
    print("TOPK docids:", topk)
    for r in refs:
        print("rerank_score:", r["score"], "chunk_id:", r["chunk_id"])
        print("content:", r["content"][:200], "...\n")

    # 7) 평가 실행
    eval_rag("../data/eval.jsonl", "sample_submission.csv")