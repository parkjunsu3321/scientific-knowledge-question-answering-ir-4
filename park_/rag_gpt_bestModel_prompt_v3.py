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

# ✅ 다국어 임베딩 모델 (E5 Large Instruct)
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = int(embed_model.get_sentence_embedding_dimension())  # ✅ dims 자동 감지

# ✅ reranker 모델 (Gemma)
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
# 1) Multilingual Embedding (E5 포맷)
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

def get_embeddings_in_batches(docs: List[Dict[str, Any]], batch_size: int = 64) -> List[List[float]]:
    """
    docs: [{"docid":..., "content":...}, ...]
    """
    all_vecs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        passages = [doc["content"] for doc in batch]
        vecs = get_embedding_passages(passages)
        all_vecs.extend(vecs)
        print(f"embedding batch {i} ~ {i+len(batch)}")
    return all_vecs


# =========================
# 2) ES 인덱스/색인 (문서 단위)
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
# 3) 검색 (BM25 + KNN 후보 생성)
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

def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 800):
    query_vec = get_embedding_query(query_str)
    knn = {
        "field": "embeddings",
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=ES_INDEX, knn=knn)

def merge_hits(bm25_hits, knn_hits, limit: int = 200) -> List[Dict[str, Any]]:
    """
    문서 단위 dedupe: docid 기준으로 합침 (chunk_id 없음)
    """
    merged = {}
    for h in bm25_hits:
        docid = h["_source"].get("docid")
        if docid and docid not in merged:
            merged[docid] = h
    for h in knn_hits:
        docid = h["_source"].get("docid")
        if docid and docid not in merged:
            merged[docid] = h

    cand = list(merged.values())
    cand.sort(key=lambda x: (x.get("_score", 0.0)), reverse=True)
    return cand[:limit]


# =========================
# 4) Reranker (CrossEncoder)
# =========================
def rerank(query: str, hits: List[Dict[str, Any]], topn: int = 30) -> List[Tuple[float, Dict[str, Any]]]:
    pairs = [(query, h["_source"]["content"]) for h in hits]
    if not pairs:
        return []

    scores = reranker.predict(pairs)
    scored = list(zip(scores.tolist(), hits))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]

def select_topk_docs(scored_hits: List[Tuple[float, Dict[str, Any]]], k_doc: int = 10) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    rerank 상위 문서 k개를 Reference로 사용
    """
    top_docids = []
    references = []
    for score, hit in scored_hits[:k_doc]:
        src = hit["_source"]
        docid = src.get("docid")
        if docid is None:
            continue
        top_docids.append(docid)
        references.append({
            "score": float(score),
            "docid": docid,
            "content": src.get("content", "")
        })
    return top_docids, references


# =========================
# 5) LLM 프롬프트/도구 정의
# =========================
persona_qa = """
## Role: 문맥 기반(Context-Aware) 지식 답변 전문가

## Instructions
당신은 제공된 [Reference] 정보를 바탕으로 사용자의 질문에 답변해야 합니다. 아래 지침을 엄격히 따르십시오.

1. **Grounding (근거 기반):** 오직 제공된 [Reference] 내의 내용에만 기반하여 답변을 작성하십시오. 당신의 배경 지식보다 [Reference]의 정보가 우선합니다.
2. **Hallucination 방지:** [Reference]에 정답을 찾을 수 없거나 정보가 부족할 경우, 솔직하게 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하십시오. 내용을 지어내지 마십시오.
3. **Style:** 답변은 한국어로 작성하며, 전문적이면서도 이해하기 쉬운 어조를 유지하십시오.
4. **Format:** 핵심 내용을 먼저 요약하고, 필요한 경우 부가 설명을 덧붙이는 두괄식으로 작성하십시오.
"""

persona_function_calling = """
## Role: 검색 의도 파악 및 쿼리 생성 전문가

## Goal
사용자의 마지막 발화와 대화 맥락을 분석하여, '외부 지식 검색'이 필요한지 판단하고 적절한 도구(function)를 호출하거나 바로 답변한다.

## Rules
1. **Search Tool 호출이 필요한 경우 (지식 질문):**
   - 사용자가 과학 상식, 역사, 특정 사실, 정의, 원리 등 **구체적인 정보**를 묻는 경우.
   - **Action:** `search` 함수를 호출하되, 대화 맥락을 고려하여 검색에 최적화된 `standalone_query`를 인자로 넘긴다.

2. **Search Tool 호출이 불필요한 경우 (일상 대화):**
   - 단순한 인사, 안부, 감정 표현, AI의 정체성 질문 등.
   - **Action:** 함수를 호출하지 않고, 사용자에게 친절하고 적절한 답변을 바로 생성한다.
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
# 6) RAG 파이프라인 (Hybrid + Rerank)
# =========================
def hybrid_search_with_rerank(query: str, k_final: int = 10, bm25_k: int = 50, knn_k: int = 50) -> Tuple[List[str], List[Dict[str, Any]]]:
    bm25 = sparse_retrieve(query, size=bm25_k)
    knn = dense_retrieve(query, size=knn_k)

    bm25_hits = bm25["hits"]["hits"]
    knn_hits = knn["hits"]["hits"]

    candidates = merge_hits(bm25_hits, knn_hits, limit=200)
    reranked = rerank(query, candidates, topn=max(60, k_final * 6))

    topk_docids, references = select_topk_docs(reranked, k_doc=k_final)
    return topk_docids, references


def answer_question(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages

    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception:
        traceback.print_exc()
        return response

    # ✅ LLM이 검색 툴 호출을 결정한 경우에만 검색/QA 수행
    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        if tool_call.function.name != "search":
            # 알 수 없는 툴 호출이면 빈 값
            return response

        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query", "").strip()
        response["standalone_query"] = standalone_query

        print(f"DEBUG: 검색 감지 -> 쿼리: {standalone_query}")

        # 검색
        topk_docids, references = hybrid_search_with_rerank(
            standalone_query, k_final=10, bm25_k=50, knn_k=50
        )
        response["topk"] = topk_docids
        response["references"] = references

        # QA
        ref_payload = [
            {"docid": r["docid"], "score": r["score"], "content": r["content"]}
            for r in references
        ]
        ref_text = json.dumps(ref_payload, ensure_ascii=False)

        qa_messages = [
            {"role": "system", "content": persona_qa},
            {"role": "system", "content": f"[Reference]\n{ref_text}"},
        ] + messages

        try:
            qaresult = client.chat.completions.create(
                model=llm_model,
                messages=qa_messages,
                temperature=0,
                seed=1,
                timeout=30
            )
            response["answer"] = qaresult.choices[0].message.content
        except Exception:
            traceback.print_exc()
            return response

        return response

    # ✅ 일상 대화면: 검색 X, API 추가 호출 X, 그냥 빈 값 반환
    print("DEBUG: 일상 대화 감지 -> output empty (no search, no QA)")
    response["answer"] = ""          # <- 빈 문자열
    response["standalone_query"] = ""
    response["topk"] = []
    response["references"] = []
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
# 7) 실행부: (재)색인 + 평가
# =========================
if __name__ == "__main__":
    # 1) 인덱스 생성
    create_es_index(ES_INDEX, settings, mappings)

    # 2) 원문 로드 (✅ 청크 없이 원문 그대로)
    with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    # raw_docs는 최소 {"docid":..., "content":...} 구조라고 가정
    docs = [{"docid": d.get("docid"), "content": d.get("content", "")} for d in raw_docs]
    docs = [d for d in docs if d["docid"] is not None and d["content"].strip()]

    print(f"raw docs: {len(docs)} (no chunking)")

    # 3) ✅ 문서 단위 임베딩 생성
    embeddings = get_embeddings_in_batches(docs, batch_size=64)

    # 4) ES 색인용 문서 구성
    index_docs = []
    for doc, emb in zip(docs, embeddings):
        doc["embeddings"] = emb
        index_docs.append(doc)

    # 5) bulk indexing
    ret = bulk_add(ES_INDEX, index_docs)
    print(ret)

    # (선택) 간단 검색 테스트
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    topk, refs = hybrid_search_with_rerank(test_query, k_final=10, bm25_k=20, knn_k=20)
    print("TOPK docids:", topk)
    for r in refs[:3]:
        print("rerank_score:", r["score"], "docid:", r["docid"])
        print("content:", r["content"][:200], "...\n")

    # 6) 평가 실행
    eval_rag("../data/eval.jsonl", "sample_submission.csv")