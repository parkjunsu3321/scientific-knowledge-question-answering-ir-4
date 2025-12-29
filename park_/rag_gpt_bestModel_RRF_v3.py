import os
import json
import re
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

# ✅ 다국어 임베딩 모델
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = int(embed_model.get_sentence_embedding_dimension())

# ✅ reranker 모델 (CrossEncoder)
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
# 1) E5 Embedding (query/passage prefix)
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
# 2) ES 인덱스/색인 (✅ 문서 단위: chunk_id 제거)
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
# 3) Retrieval (BM25 + KNN) + RRF
# =========================
def sparse_retrieve(query_str: str, size: int = 50):
    # ✅ BM25 precision 강화 (노이즈 감소)
    query = {
        "match": {
            "content": {
                "query": query_str,
                "operator": "and",
                "minimum_should_match": "50%"
            }
        }
    }
    return es.search(index=ES_INDEX, query=query, size=size)

def dense_retrieve(query_str: str, size: int = 50, num_candidates: int = 800):
    query_vec = get_embedding_query(query_str)
    knn = {
        "field": "embeddings",
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates
    }
    return es.search(index=ES_INDEX, knn=knn)

def _docid_from_hit(hit: Dict[str, Any]) -> str:
    src = hit.get("_source", {})
    return str(src.get("docid", ""))

def rrf_merge_hits(
    bm25_hits: List[Dict[str, Any]],
    knn_hits: List[Dict[str, Any]],
    limit: int = 120,
    k: int = 60,
    w_bm25: float = 1.0,
    w_knn: float = 1.0
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    hit_map: Dict[str, Dict[str, Any]] = {}

    for i, h in enumerate(bm25_hits):
        did = _docid_from_hit(h)
        if not did:
            continue
        hit_map[did] = h
        rank = i + 1
        scores[did] = scores.get(did, 0.0) + (w_bm25 / (k + rank))

    for i, h in enumerate(knn_hits):
        did = _docid_from_hit(h)
        if not did:
            continue
        hit_map.setdefault(did, h)
        rank = i + 1
        scores[did] = scores.get(did, 0.0) + (w_knn / (k + rank))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    merged_hits = []
    for did, rrf_score in ranked[:limit]:
        h = hit_map[did]
        h["_rrf_score"] = float(rrf_score)
        merged_hits.append(h)

    return merged_hits


def build_evidence_text(doc_text: str, max_len: int = 600) -> str:
    """
    문서가 짧아도(<=1200자) reranker는 '핵심 부분'을 주면 더 안정적일 때가 많음.
    - 문서 전체가 max_len 이하면 그대로
    - 길면 head/mid/tail 섞어서 추출
    """
    if not doc_text:
        return ""
    t = doc_text.strip()
    if len(t) <= max_len:
        return t

    head = t[:250]
    mid_start = max(0, len(t)//2 - 150)
    mid = t[mid_start:mid_start + 300]
    tail = t[-250:]
    return f"{head}\n{mid}\n{tail}"[:max_len]


def rerank(query: str, hits: List[Dict[str, Any]], topn: int = 40) -> List[Tuple[float, Dict[str, Any]]]:
    if not hits:
        return []

    pairs = []
    enriched = []
    for h in hits:
        content = h["_source"].get("content", "")
        evidence = build_evidence_text(content, max_len=600)
        h["_source"]["evidence"] = evidence  # QA에 evidence를 줄 수도 있게 저장
        pairs.append((query, evidence))
        enriched.append(h)

    scores = reranker.predict(pairs)
    scored = list(zip(scores.tolist(), enriched))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]


def select_topk_docs(scored_hits: List[Tuple[float, Dict[str, Any]]], k_doc: int = 3) -> Tuple[List[str], List[Dict[str, Any]]]:
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
            # ✅ QA에는 evidence를 주는 것이 더 좋음(짧고 정확)
            "content": src.get("evidence") or src.get("content", "")
        })
    return top_docids, references


# =========================
# 5) LLM Router + QA Prompt
# =========================
persona_router = """
## Role: 지식 검색 전문가

## Instruction
- 사용자가 어떤 주제에 대해 질문하면 **무조건 문서 검색을 최우선**으로 수행해야 한다. (당신의 지식을 절대 과신하지 마세요.)
- 제공된 문서 데이터베이스에는 과학, 역사, 문화, 사회, 인물, 기술 등 다양한 주제의 지식이 포함되어 있다.
- **검색어 생성 원칙 (가장 중요):**
  1. 검색어(`standalone_query`)는 반드시 **한국어** 위주로 생성한다.
  2. 질문에 영어 고유명사나 전문 용어가 포함된 경우, 반드시 한글 번역어와 원어를 함께 넣어 검색하라.
  3. 문장 형태가 아닌 검색 엔진이 이해하기 쉬운 **핵심 키워드 나열 방식**을 사용한다.
  4. 대화 맥락을 파악하여 대명사(그것, 그게, 이것 등)나 생략된 주어를 구체적인 대상으로 치환한다.
- **지식 질문 판별:**
  - 사실/개념/정보/설명 요구면 `needs_search=true`
  - 단순 인사/감사/잡담만 `needs_search=false`
- **출력 형식:** JSON만 출력한다.
  - `needs_search`: true/false
  - `standalone_query`: 검색 쿼리 (지식 질문일 때만)
  - `brief_reply`: 단순 잡담일 때만(여기서는 사용 안 해도 됨)
"""

# ✅ “없음” 남발 방지: 부분 정보라도 있으면 그 범위 내에서 답변
persona_qa = """
## Role: 신뢰도 높은 지식 답변가

## Instruction
- 답변은 **반드시** 제공된 [Reference]에 근거한다.
- [Reference]에 명시된 정보만 사용한다. 추측/창작 금지.
- 다만, [Reference]에 일부 정보라도 관련이 있으면 **그 범위 내에서** 최대한 설명한다.
- 정말로 관련 정보가 없을 때만: "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변한다.
- 한국어로 간결하게 작성한다.
"""

def _safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*```$", "", t).strip()
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            t = m.group(0)
    try:
        return json.loads(t)
    except Exception:
        return {}

def route(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    out = {"needs_search": True, "standalone_query": "", "brief_reply": ""}
    try:
        r = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": persona_router}] + messages,
            temperature=0,
            seed=1,
            timeout=10
        )
        j = _safe_json_loads(r.choices[0].message.content or "")
        if isinstance(j.get("needs_search"), bool):
            out["needs_search"] = j["needs_search"]
        if isinstance(j.get("standalone_query"), str):
            out["standalone_query"] = j["standalone_query"].strip()
        if isinstance(j.get("brief_reply"), str):
            out["brief_reply"] = j["brief_reply"].strip()
        return out
    except Exception:
        traceback.print_exc()
        return out


# =========================
# 6) RAG 파이프라인 (Hybrid + RRF + Rerank)
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

    candidates = rrf_merge_hits(
        bm25_hits=bm25_hits,
        knn_hits=knn_hits,
        limit=rrf_limit,
        k=rrf_k,
        w_bm25=w_bm25,
        w_knn=w_knn
    )

    reranked = rerank(query, candidates, topn=max(40, k_final * 10))
    topk_docids, references = select_topk_docs(reranked, k_doc=k_final)
    return topk_docids, references


def answer_question(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    ✅ 일상 대화면: 검색 X, QA X, answer="" 반환
    ✅ 지식 질문이면: 검색 + QA 수행
    """
    response = {"needs_search": True, "standalone_query": "", "topk": [], "references": [], "answer": ""}

    routing = route(messages)
    needs_search = bool(routing.get("needs_search", True))
    response["needs_search"] = needs_search

    if not needs_search:
        # ✅ 요구사항: 일상 대화는 검색도 API도 하지 말고 비워주기
        # (라우터 1회 호출은 unavoidable. 그 외 검색/QA는 안 함)
        response["answer"] = ""
        return response

    standalone_query = (routing.get("standalone_query") or "").strip()
    if not standalone_query:
        # fallback: 마지막 user 발화로 검색
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        standalone_query = (last_user or "").strip()

    response["standalone_query"] = standalone_query

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
    response["topk"] = topk_docids
    response["references"] = references

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
        response["answer"] = qaresult.choices[0].message.content or ""
    except Exception:
        traceback.print_exc()
        return response

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
                "needs_search": response.get("needs_search", True),
                "standalone_query": response.get("standalone_query", ""),
                "topk": response.get("topk", []),
                "answer": response.get("answer", ""),
                "references": response.get("references", [])
            }
            of.write(f"{json.dumps(output, ensure_ascii=False)}\n")
            idx += 1


# =========================
# 7) 실행부: (재)색인 + 평가
# =========================
if __name__ == "__main__":
    # ✅ 문서 길이 1200자 이하 전제: chunking 제거(문서 단위 색인)
    create_es_index(ES_INDEX, settings, mappings)

    with open("../data/documents.jsonl", "r", encoding="utf-8") as f:
        raw_docs = [json.loads(line) for line in f]

    docs = [{"docid": d.get("docid"), "content": d.get("content", "")} for d in raw_docs]
    docs = [d for d in docs if d["docid"] is not None and d["content"].strip()]

    print(f"raw docs: {len(docs)} (no chunking)")

    embeddings = get_embeddings_in_batches(docs, batch_size=128)

    index_docs = []
    for doc, emb in zip(docs, embeddings):
        doc["embeddings"] = emb
        index_docs.append(doc)

    ret = bulk_add(ES_INDEX, index_docs)
    print(ret)

    # (선택) 간단 검색 테스트
    test_query = "금성이 다른 행성들보다 밝게 보이는 이유는 무엇인가요?"
    topk, refs = hybrid_search_with_rerank(test_query, k_final=3, bm25_k=20, knn_k=20)
    print("TOPK docids:", topk)
    for r in refs:
        print("rerank_score:", r["score"], "docid:", r["docid"])
        print("content:", r["content"][:200], "...\n")

    # 평가
    eval_rag("../data/eval.jsonl", "sample_submission.csv")