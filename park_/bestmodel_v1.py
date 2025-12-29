import os
import json
import hashlib
import traceback
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI


# =========================================================
# 0) 환경/모델 설정
# =========================================================
_ = load_dotenv()

# ✅ OpenAI
client = OpenAI()
llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ✅ Embedding / Reranker
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large-instruct")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = int(embed_model.get_sentence_embedding_dimension())
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# ✅ Elasticsearch
ES_INDEX = os.getenv("ES_INDEX", "test")
ES_URL = os.getenv("ES_URL", "https://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "x9lxBdj3-1O-yi8+4sFA")
ES_CA_CERTS = os.getenv("ES_CA_CERTS", "/opt/elasticsearch-8.8.0/config/certs/http_ca.crt")

es = Elasticsearch(
    [ES_URL],
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs=ES_CA_CERTS,
    request_timeout=60,
    retry_on_timeout=True,
    max_retries=3,
)

print(es.info())


# =========================================================
# 1) E5 Embedding
# =========================================================
def e5_passage(text: str) -> str:
    return f"passage: {text}"

def e5_query(text: str) -> str:
    return f"query: {text}"

def get_embedding_passages(passages: List[str]) -> List[List[float]]:
    vecs = embed_model.encode(
        [e5_passage(p) for p in passages],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()

def get_embedding_query(query: str) -> List[float]:
    vec = embed_model.encode(
        [e5_query(query)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vec.tolist()

def get_embeddings_in_batches(docs: List[Dict[str, Any]], batch_size: int = 64) -> List[List[float]]:
    all_vecs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        passages = [doc["content"] for doc in batch]
        vecs = get_embedding_passages(passages)
        all_vecs.extend(vecs)
        print(f"[EMB] batch {i} ~ {i + len(batch)}")
    return all_vecs


# =========================================================
# 2) ES 인덱스/색인
# =========================================================
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

def bulk_add(index: str, docs: List[Dict[str, Any]], chunk_size: int = 500) -> Any:
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions, chunk_size=chunk_size, request_timeout=120)


# =========================================================
# 3) 검색 (BM25 + KNN) + RRF
# =========================================================
_query_emb_cache: Dict[str, List[float]] = {}

def get_embedding_query_cached(query: str) -> List[float]:
    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    if key in _query_emb_cache:
        return _query_emb_cache[key]
    vec = get_embedding_query(query)
    _query_emb_cache[key] = vec
    return vec

def sparse_retrieve(query_str: str, size: int = 50):
    query = {"match": {"content": {"query": query_str}}}
    return es.search(index=ES_INDEX, query=query, size=size, sort="_score")

def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 800):
    query_vec = get_embedding_query_cached(query_str)
    knn = {
        "field": "embeddings",
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=ES_INDEX, knn=knn)

def merge_hits_rrf(
    bm25_hits: List[Dict[str, Any]],
    knn_hits: List[Dict[str, Any]],
    limit: int = 200,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    best_hit: Dict[str, Dict[str, Any]] = {}

    def add_rrf(hits: List[Dict[str, Any]]):
        for rank, h in enumerate(hits, start=1):
            docid = h.get("_source", {}).get("docid")
            if not docid:
                continue
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank)
            if docid not in best_hit or h.get("_score", 0.0) > best_hit[docid].get("_score", 0.0):
                best_hit[docid] = h

    add_rrf(bm25_hits)
    add_rrf(knn_hits)

    merged = []
    for docid, h in best_hit.items():
        h = dict(h)
        h["_rrf_score"] = float(scores.get(docid, 0.0))
        merged.append(h)

    merged.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return merged[:limit]


# =========================================================
# 4) Reranker
# =========================================================
def rerank(query: str, hits: List[Dict[str, Any]], topn: int = 60) -> List[Tuple[float, Dict[str, Any]]]:
    pairs = []
    filtered = []
    for h in hits:
        src = h.get("_source", {})
        content = (src.get("content") or "").strip()
        if not content:
            continue
        if len(content) > 2000:
            content = content[:2000]
            h = dict(h)
            h["_source"] = dict(src)
            h["_source"]["content"] = content
        pairs.append((query, content))
        filtered.append(h)

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    scored = list(zip(scores.tolist(), filtered))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]

def select_topk_docs(scored_hits: List[Tuple[float, Dict[str, Any]]], k_doc: int = 10) -> Tuple[List[str], List[Dict[str, Any]]]:
    top_docids = []
    references = []
    for score, hit in scored_hits[:k_doc]:
        src = hit.get("_source", {})
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


# =========================================================
# 5) 프롬프트
# =========================================================
persona_function_calling = """
## Role: 검색 의도 판별 및 '최적 검색쿼리' 생성 전문가

- 답변 생성 금지.
- 지식/사실/정의/원리/설명/근거/비교 요청이면 search 호출.
- 단순 인사/감정표현/짧은 리액션이면 호출하지 않음.

standalone_query는 짧고 핵심 키워드 중심으로 작성.
"""

persona_qa = """
## Role: 문서 근거 기반(RAG Grounded) QA 전문가

- [Reference]만 근거로 답변.
- 근거 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변.
- 문장 끝에 [docid]로 근거 표기.
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"]
            }
        }
    },
]


# =========================================================
# 6) 유틸
# =========================================================
def format_references_for_prompt(references: List[Dict[str, Any]]) -> str:
    lines = []
    for r in references:
        docid = r.get("docid")
        score = r.get("score")
        content = (r.get("content") or "").strip()
        lines.append(f"- docid: {docid}\n  score: {score}\n  content: {content}")
    return "\n".join(lines)

def normalize_messages(msg: Any) -> List[Dict[str, str]]:
    if isinstance(msg, str):
        return [{"role": "user", "content": msg}]
    if isinstance(msg, list):
        out = []
        for m in msg:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": m["role"], "content": m["content"]})
        if out:
            return out
    return [{"role": "user", "content": str(msg)}]


# =========================================================
# 7) ✅ 일상대화 판별 (단, 제출 topk는 항상 리스트)
# =========================================================
def is_smalltalk(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    if len(t) <= 3:
        return True

    smalltalk_patterns = [
        "안녕", "하이", "hello", "hi",
        "고마워", "감사", "thanks", "thank you",
        "ㅇㅋ", "오케이", "ok",
        "ㅋㅋ", "ㅎㅎ", "lol",
        "좋아", "굿", "굳", "nice",
        "수고", "수고했어",
    ]
    return any(p in t for p in smalltalk_patterns)


# =========================================================
# 8) Hybrid Search + Rerank
# =========================================================
def hybrid_search_with_rerank(
    query: str,
    k_final: int = 10,
    bm25_k: int = 50,
    knn_k: int = 50,
    num_candidates: int = 800,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    bm25 = sparse_retrieve(query, size=bm25_k)
    knn = dense_retrieve(query, size=knn_k, num_candidates=num_candidates)

    bm25_hits = bm25["hits"]["hits"]
    knn_hits = knn["hits"]["hits"]

    candidates = merge_hits_rrf(bm25_hits, knn_hits, limit=200, rrf_k=60)
    reranked = rerank(query, candidates, topn=max(60, k_final * 6))

    topk_docids, references = select_topk_docs(reranked, k_doc=k_final)
    return topk_docids, references


# =========================================================
# 9) Answer
#   ✅ 핵심 수정:
#   - 절대 topk=None 금지
#   - smalltalk / no toolcall 등에서도 topk는 []
# =========================================================
def answer_question(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    # ✅ 기본값: 평가 코드 안전하게 통과
    response = {
        "standalone_query": "",
        "topk": [],          # ✅ 절대 None 금지
        "references": [],    # ✅ list
        "answer": "",        # ✅ string
    }

    # 마지막 user 추출
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "")
            break

    # ✅ 일상 대화면: search/QA 호출 없이 기본값 그대로 반환
    if is_smalltalk(last_user_msg):
        print("[DEBUG] smalltalk -> skip search & QA, but keep topk=[] for evaluation safety")
        return response

    # 1) function-calling
    fc_messages = [{"role": "system", "content": persona_function_calling}] + messages

    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=fc_messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            seed=1,
            timeout=15
        )
    except Exception:
        traceback.print_exc()
        return response

    tool_calls = result.choices[0].message.tool_calls
    if not tool_calls:
        # ✅ tool-call 없으면 검색도 QA도 안 함 (하지만 topk는 [])
        print("[DEBUG] no tool-call -> skip search & QA (topk stays [])")
        return response

    tool_call = tool_calls[0]
    if tool_call.function.name != "search":
        print("[DEBUG] unknown tool-call -> skip")
        return response

    try:
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = (function_args.get("standalone_query") or "").strip()
    except Exception:
        standalone_query = ""

    if not standalone_query:
        print("[DEBUG] empty standalone_query -> skip")
        return response

    response["standalone_query"] = standalone_query
    print(f"[DEBUG] search query -> {standalone_query}")

    # 2) Search + Rerank
    try:
        topk_docids, references = hybrid_search_with_rerank(
            standalone_query, k_final=10, bm25_k=50, knn_k=50, num_candidates=800
        )
    except Exception:
        traceback.print_exc()
        return response

    response["topk"] = topk_docids or []
    response["references"] = references or []

    # 3) QA (Grounded)
    ref_text = format_references_for_prompt(response["references"])
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
            timeout=45
        )
        response["answer"] = qaresult.choices[0].message.content or ""
    except Exception:
        traceback.print_exc()
        return response

    return response


# =========================================================
# 10) Eval: jsonl 내용 + 파일명만 csv
# =========================================================
def eval_rag(eval_filename: str, output_filename: str):
    with open(eval_filename, "r", encoding="utf-8") as f, open(output_filename, "w", encoding="utf-8") as of:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                j = json.loads(line)
            except Exception:
                print(f"[WARN] invalid json at line {idx}")
                continue

            msgs = normalize_messages(j.get("msg"))
            response = answer_question(msgs)

            # ✅ 반드시 topk는 list로
            if response.get("topk") is None:
                response["topk"] = []

            output = {
                "eval_id": j.get("eval_id"),
                "standalone_query": response.get("standalone_query", ""),
                "topk": response.get("topk", []),
                "answer": response.get("answer", ""),
                "references": response.get("references", []),
            }
            of.write(json.dumps(output, ensure_ascii=False) + "\n")


# =========================================================
# 11) 실행부
# =========================================================
if __name__ == "__main__":
    # 1) 인덱스 생성
    create_es_index(ES_INDEX, settings, mappings)

    # 2) 원문 로드
    with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f if line.strip()]

    docs = [{"docid": d.get("docid"), "content": d.get("content", "")} for d in raw_docs]
    docs = [d for d in docs if d["docid"] is not None and (d["content"] or "").strip()]

    print(f"[DATA] raw docs: {len(docs)} (no chunking)")

    # 3) 임베딩 생성
    embeddings = get_embeddings_in_batches(docs, batch_size=64)

    # 4) 색인 문서 구성
    index_docs = []
    for doc, emb in zip(docs, embeddings):
        dd = dict(doc)
        dd["embeddings"] = emb
        index_docs.append(dd)

    # 5) bulk indexing
    ret = bulk_add(ES_INDEX, index_docs, chunk_size=500)
    print("[INDEX] bulk result:", ret)

    # 6) 평가 실행 (내용은 jsonl, 파일명만 csv)
    eval_rag("../data/eval.jsonl", "sample_submission.csv")