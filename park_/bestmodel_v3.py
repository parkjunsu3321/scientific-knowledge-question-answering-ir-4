import os
import re
import json
import hashlib
import traceback
from typing import List, Dict, Any, Tuple, Optional

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
ES_INDEX = os.getenv("ES_INDEX", "test_bilingual")
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
# 1) E5 Embedding (멀티링구얼이라 ko/en 모두 동일 모델로 OK)
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


def get_embeddings_in_batches_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = get_embedding_passages(batch)
        all_vecs.extend(vecs)
        print(f"[EMB] batch {i} ~ {i + len(batch)}")
    return all_vecs


# =========================================================
# 2) ES 인덱스/색인 (KO/EN 필드 + 각각 dense_vector)
# =========================================================
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"],
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"],
            }
        },
    }
}

mappings = {
    "properties": {
        "docid": {"type": "keyword"},

        # KO text (nori)
        "content_ko": {"type": "text", "analyzer": "nori"},
        "embeddings_ko": {
            "type": "dense_vector",
            "dims": EMBED_DIM,
            "index": True,
            "similarity": "cosine",
        },

        # EN text (standard)
        "content_en": {"type": "text"},
        "embeddings_en": {
            "type": "dense_vector",
            "dims": EMBED_DIM,
            "index": True,
            "similarity": "cosine",
        },
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
# 3) 검색 (BM25 + KNN) + RRF (KO/EN 4개 결과를 한 번에 앙상블)
# =========================================================
_query_emb_cache: Dict[str, List[float]] = {}


def get_embedding_query_cached(query: str) -> List[float]:
    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    if key in _query_emb_cache:
        return _query_emb_cache[key]
    vec = get_embedding_query(query)
    _query_emb_cache[key] = vec
    return vec


def sparse_retrieve(query_str: str, field: str, size: int = 50):
    query = {"match": {field: {"query": query_str}}}
    return es.search(index=ES_INDEX, query=query, size=size, sort="_score")


def dense_retrieve(query_str: str, vector_field: str, size: int = 50, num_candidates: int = 800):
    query_vec = get_embedding_query_cached(query_str)
    knn = {
        "field": vector_field,
        "query_vector": query_vec,
        "k": size,
        "num_candidates": num_candidates,
    }
    return es.search(index=ES_INDEX, knn=knn)


def merge_hits_rrf_multi(
    hit_lists: List[List[Dict[str, Any]]],
    limit: int = 200,
    rrf_k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    hit_lists: [bm25_ko_hits, knn_ko_hits, bm25_en_hits, knn_en_hits] 등 여러 리스트
    weights: 각 리스트별 가중치. None이면 모두 1.0
    """
    if weights is None:
        weights = [1.0] * len(hit_lists)
    if len(weights) != len(hit_lists):
        weights = [1.0] * len(hit_lists)

    scores: Dict[str, float] = {}
    best_hit: Dict[str, Dict[str, Any]] = {}

    def add_rrf(hits: List[Dict[str, Any]], w: float):
        for rank, h in enumerate(hits, start=1):
            src = h.get("_source", {}) or {}
            docid = src.get("docid")
            if not docid:
                continue
            scores[docid] = scores.get(docid, 0.0) + w * (1.0 / (rrf_k + rank))
            # best_hit은 score 큰 걸로 보관(그냥 보기용)
            if docid not in best_hit or h.get("_score", 0.0) > best_hit[docid].get("_score", 0.0):
                best_hit[docid] = h

    for hits, w in zip(hit_lists, weights):
        add_rrf(hits, w)

    merged: List[Dict[str, Any]] = []
    for docid, h in best_hit.items():
        hh = dict(h)
        hh["_rrf_score"] = float(scores.get(docid, 0.0))
        merged.append(hh)

    merged.sort(key=lambda x: x.get("_rrf_score", 0.0), reverse=True)
    return merged[:limit]


# =========================================================
# 4) Reranker (KO/EN 쿼리 및 KO/EN 문서 내용을 같이 보고 max/avg로 결합)
# =========================================================
def rerank_bilingual(
    query_ko: str,
    query_en: str,
    hits: List[Dict[str, Any]],
    topn: int = 60,
    combine: str = "max",  # "max" or "avg"
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    각 doc에 대해:
      score_ko = reranker(query_ko, content_ko)
      score_en = reranker(query_en, content_en)
    를 구해서 combine 규칙으로 최종 점수를 만든다.
    """
    # 쿼리 비었으면 반대쪽만 사용
    query_ko = (query_ko or "").strip()
    query_en = (query_en or "").strip()

    # docid 단위로 content 준비
    docs: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {}) or {}
        docid = src.get("docid")
        if not docid:
            continue
        content_ko = (src.get("content_ko") or "").strip()
        content_en = (src.get("content_en") or "").strip()
        if not content_ko and not content_en:
            continue
        docs.append(h)

    if not docs:
        return []

    # reranker 입력 pairs 구성 (쿼리/콘텐츠 둘 다 있을 때만)
    pairs_ko = []
    idx_ko = []
    if query_ko:
        for i, h in enumerate(docs):
            c = ((h.get("_source", {}) or {}).get("content_ko") or "").strip()
            if c:
                pairs_ko.append((query_ko, c))
                idx_ko.append(i)

    pairs_en = []
    idx_en = []
    if query_en:
        for i, h in enumerate(docs):
            c = ((h.get("_source", {}) or {}).get("content_en") or "").strip()
            if c:
                pairs_en.append((query_en, c))
                idx_en.append(i)

    # 점수 배열(문서별)
    s_ko = [None] * len(docs)
    s_en = [None] * len(docs)

    try:
        if pairs_ko:
            sc = reranker.predict(pairs_ko).tolist()
            for score, i in zip(sc, idx_ko):
                s_ko[i] = float(score)
    except Exception:
        traceback.print_exc()

    try:
        if pairs_en:
            sc = reranker.predict(pairs_en).tolist()
            for score, i in zip(sc, idx_en):
                s_en[i] = float(score)
    except Exception:
        traceback.print_exc()

    # 결합 점수
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for i, h in enumerate(docs):
        a = s_ko[i]
        b = s_en[i]
        vals = [v for v in [a, b] if v is not None]
        if not vals:
            continue
        if combine.lower() == "avg":
            final = sum(vals) / len(vals)
        else:
            final = max(vals)
        scored.append((float(final), h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topn]


def select_topk_docs(
    scored_hits: List[Tuple[float, Dict[str, Any]]],
    k_doc: int = 10,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    top_docids: List[str] = []
    references: List[Dict[str, Any]] = []
    for score, hit in scored_hits[:k_doc]:
        src = hit.get("_source", {}) or {}
        docid = src.get("docid")
        if docid is None:
            continue
        top_docids.append(docid)
        references.append(
            {
                "score": float(score),
                "docid": docid,
                # 디버그/QA용으로 KO/EN 둘 다 넣어둠
                "content_ko": src.get("content_ko", ""),
                "content_en": src.get("content_en", ""),
            }
        )
    return top_docids, references


# =========================================================
# 5) LLM 기반 일상대화 판별
# =========================================================
SMALLTALK_GUARD_SYSTEM = """
You are a strict classifier for Korean user messages.

Task:
- Decide if the user's last message is "SMALLTALK" (casual chat, greetings, thanks, emotions, reactions, chit-chat)
  OR "KNOWLEDGE" (asking for facts, definitions, explanations, comparisons, effects/impacts, how-to, troubleshooting, analysis).

Rules:
- If the user asks about effects/benefits/advantages ("효과", "이점", "장점", "긍정적인 영향"), classify as KNOWLEDGE.
- If the user asks any question that could be answered with documents or search, classify as KNOWLEDGE.
- Only mark SMALLTALK when it is clearly not a knowledge request.

Output (MUST be valid JSON, nothing else):
{"label":"SMALLTALK"} or {"label":"KNOWLEDGE"}
""".strip()


def llm_is_smalltalk(last_user_text: str, timeout: int = 10) -> bool:
    if not last_user_text or not last_user_text.strip():
        return True

    try:
        r = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": SMALLTALK_GUARD_SYSTEM},
                {"role": "user", "content": last_user_text.strip()},
            ],
            temperature=0,
            seed=1,
            timeout=timeout,
        )
        content = (r.choices[0].message.content or "").strip()
        j = json.loads(content)
        label = (j.get("label") or "").strip().upper()
        return label == "SMALLTALK"
    except Exception:
        traceback.print_exc()
        return False


# =========================================================
# 6) 프롬프트 (function calling)
# =========================================================
persona_function_calling = """
## Role: 검색 의도 판별 및 '최적 검색쿼리' 생성 전문가
- 답변 생성 금지.
- 지식/사실/정의/원리/설명/근거/비교/효과/장단점/이유/영향 요청이면 반드시 search 호출.
- 단순 인사/감정표현/짧은 리액션이면 호출하지 않음.
- 스스로의 지식으로 답할 수 있어도, 지식 질문이면 반드시 search를 먼저 호출한다.
standalone_query는 짧고 핵심 키워드 중심으로 작성.
""".strip()

persona_qa = """
## Role: 문서 근거 기반(RAG Grounded) QA 전문가
- [Reference]만 근거로 답변.
- 근거 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변.
- 문장 끝에 [docid]로 근거 표기.
""".strip()


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
                        "description": "Final query suitable for use in search from the user messages history.",
                    }
                },
                "required": ["standalone_query"],
            },
        },
    },
]


# =========================================================
# 7) 유틸
# =========================================================
def format_references_for_prompt(references: List[Dict[str, Any]]) -> str:
    lines = []
    for r in references:
        docid = r.get("docid")
        score = r.get("score")
        cko = (r.get("content_ko") or "").strip()
        cen = (r.get("content_en") or "").strip()
        lines.append(
            f"- docid: {docid}\n"
            f"  score: {score}\n"
            f"  content_ko: {cko}\n"
            f"  content_en: {cen}"
        )
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


def get_last_user_message(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


# =========================================================
# 8) Hybrid Search + Rerank (Bilingual Ensemble)
# =========================================================
def hybrid_search_with_rerank_bilingual(
    query_ko: str,
    query_en: str,
    k_final: int = 10,
    bm25_k: int = 50,
    knn_k: int = 50,
    num_candidates: int = 800,
    rrf_k: int = 60,
    rrf_weights: Optional[List[float]] = None,  # [bm25_ko, knn_ko, bm25_en, knn_en]
    rerank_combine: str = "max",               # "max" 추천
) -> Tuple[List[str], List[Dict[str, Any]]]:
    query_ko = (query_ko or "").strip()
    query_en = (query_en or "").strip()

    # (1) KO 검색
    bm25_ko_hits: List[Dict[str, Any]] = []
    knn_ko_hits: List[Dict[str, Any]] = []
    if query_ko:
        bm25_ko = sparse_retrieve(query_ko, field="content_ko", size=bm25_k)
        knn_ko = dense_retrieve(query_ko, vector_field="embeddings_ko", size=knn_k, num_candidates=num_candidates)
        bm25_ko_hits = bm25_ko["hits"]["hits"]
        knn_ko_hits = knn_ko["hits"]["hits"]

    # (2) EN 검색
    bm25_en_hits: List[Dict[str, Any]] = []
    knn_en_hits: List[Dict[str, Any]] = []
    if query_en:
        bm25_en = sparse_retrieve(query_en, field="content_en", size=bm25_k)
        knn_en = dense_retrieve(query_en, vector_field="embeddings_en", size=knn_k, num_candidates=num_candidates)
        bm25_en_hits = bm25_en["hits"]["hits"]
        knn_en_hits = knn_en["hits"]["hits"]

    # (3) 4개 결과를 RRF로 통합
    hit_lists = [bm25_ko_hits, knn_ko_hits, bm25_en_hits, knn_en_hits]
    candidates = merge_hits_rrf_multi(
        hit_lists=hit_lists,
        limit=200,
        rrf_k=rrf_k,
        weights=rrf_weights or [1.0, 1.0, 1.0, 1.0],
    )

    # (4) Rerank: (ko_query vs ko_content), (en_query vs en_content) 점수 결합
    reranked = rerank_bilingual(
        query_ko=query_ko,
        query_en=query_en,
        hits=candidates,
        topn=max(60, k_final * 6),
        combine=rerank_combine,
    )

    # (5) topk
    topk_docids, references = select_topk_docs(reranked, k_doc=k_final)
    return topk_docids, references


# =========================================================
# 9) Answer
#   - eval_en 번역 msg를 같이 받아 query_en으로 사용
# =========================================================
def answer_question(messages_ko: List[Dict[str, str]], messages_en: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    response = {
        "standalone_query": "",
        "standalone_query_en": "",
        "topk": [],
        "references": [],
        "answer": "",
    }

    last_user_msg_ko = get_last_user_message(messages_ko)
    last_user_msg_en = get_last_user_message(messages_en) if messages_en else ""

    # ✅ 0) smalltalk 판별은 KO 기준(원문 메시지 기준)
    if llm_is_smalltalk(last_user_msg_ko):
        print("[DEBUG] LLM smalltalk=True -> skip search & QA, keep topk=[]")
        return response

    # 1) function-calling (KO에서 standalone_query 생성)
    fc_messages = [{"role": "system", "content": persona_function_calling}] + messages_ko
    standalone_query_ko = ""
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=fc_messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
            seed=1,
            timeout=15,
        )
        tool_calls = result.choices[0].message.tool_calls
    except Exception:
        traceback.print_exc()
        tool_calls = None

    if tool_calls:
        tool_call = tool_calls[0]
        if tool_call.function.name == "search":
            try:
                function_args = json.loads(tool_call.function.arguments)
                standalone_query_ko = (function_args.get("standalone_query") or "").strip()
            except Exception:
                standalone_query_ko = ""

    if not standalone_query_ko:
        standalone_query_ko = (last_user_msg_ko or "").strip()

    # EN 쿼리는 번역 msg가 있으면 그걸 그대로(또는 마지막 메시지) 사용
    standalone_query_en = (last_user_msg_en or "").strip()

    if not standalone_query_ko and not standalone_query_en:
        print("[DEBUG] empty standalone_query_ko/en -> skip")
        return response

    response["standalone_query"] = standalone_query_ko
    response["standalone_query_en"] = standalone_query_en
    print(f"[DEBUG] search query_ko -> {standalone_query_ko}")
    print(f"[DEBUG] search query_en -> {standalone_query_en}")

    # 2) Search + Rerank (Bilingual Ensemble)
    try:
        topk_docids, references = hybrid_search_with_rerank_bilingual(
            query_ko=standalone_query_ko,
            query_en=standalone_query_en,
            k_final=10,
            bm25_k=50,
            knn_k=50,
            num_candidates=800,
            rrf_k=60,
            # 필요하면 여기서 EN 쪽 가중치 더 주기 가능
            # 예: [1.0, 1.0, 1.2, 1.2]
            rrf_weights=[1.0, 1.0, 1.0, 1.0],
            rerank_combine="max",
        )
    except Exception:
        traceback.print_exc()
        return response

    response["topk"] = topk_docids or []
    response["references"] = references or []

    # 3) QA (Grounded) - messages_ko 기준 답변 생성 (참조는 ko/en 둘 다 제공)
    ref_text = format_references_for_prompt(response["references"])
    qa_messages = [
        {"role": "system", "content": persona_qa},
        {"role": "system", "content": f"[Reference]\n{ref_text}"},
    ] + messages_ko

    try:
        qaresult = client.chat.completions.create(
            model=llm_model,
            messages=qa_messages,
            temperature=0,
            seed=1,
            timeout=45,
        )
        response["answer"] = qaresult.choices[0].message.content or ""
    except Exception:
        traceback.print_exc()
        return response

    return response


# =========================================================
# 10) Eval: KO/EN eval.jsonl을 eval_id로 매칭해서 앙상블
# =========================================================
def load_eval_map(eval_filename: str) -> Dict[str, Any]:
    """
    eval.jsonl -> { eval_id: raw_json }
    """
    m: Dict[str, Any] = {}
    with open(eval_filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                print(f"[WARN] invalid json at line {idx} in {eval_filename}")
                continue
            eid = str(j.get("eval_id"))
            if eid:
                m[eid] = j
    return m


def eval_rag_bilingual(eval_ko_filename: str, eval_en_filename: str, output_filename: str):
    eval_ko = load_eval_map(eval_ko_filename)
    eval_en = load_eval_map(eval_en_filename) if eval_en_filename and os.path.exists(eval_en_filename) else {}

    with open(output_filename, "w", encoding="utf-8") as of:
        for eid, row_ko in eval_ko.items():
            msgs_ko = normalize_messages(row_ko.get("msg"))
            row_en = eval_en.get(eid)
            msgs_en = normalize_messages(row_en.get("msg")) if row_en else None

            response = answer_question(msgs_ko, msgs_en)

            if response.get("topk") is None:
                response["topk"] = []

            output = {
                "eval_id": row_ko.get("eval_id"),
                "standalone_query": response.get("standalone_query", ""),
                "standalone_query_en": response.get("standalone_query_en", ""),
                "topk": response.get("topk", []),
                "answer": response.get("answer", ""),
                "references": response.get("references", []),
            }
            of.write(json.dumps(output, ensure_ascii=False) + "\n")


# =========================================================
# 11) 실행부
# =========================================================
if __name__ == "__main__":
    # 파일 경로 (네 데이터에 맞게 조정)
    DOCS_KO_PATH = os.getenv("DOCS_KO_PATH", "../data/documents.jsonl")
    DOCS_EN_PATH = os.getenv("DOCS_EN_PATH", "../data/documents_en.jsonl")
    EVAL_KO_PATH = os.getenv("EVAL_KO_PATH", "../data/eval.jsonl")
    EVAL_EN_PATH = os.getenv("EVAL_EN_PATH", "../data/eval_en.jsonl")
    OUT_JSONL_PATH = os.getenv("OUT_JSONL_PATH", "sample_submission.csv")

    # 1) 인덱스 생성
    create_es_index(ES_INDEX, settings, mappings)

    # 2) 원문 로드 (ko/en을 docid로 조인)
    docs_ko_map: Dict[str, str] = {}
    with open(DOCS_KO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            docid = str(j.get("docid"))
            content = (j.get("content") or "").strip()
            if docid and content:
                docs_ko_map[docid] = content

    docs_en_map: Dict[str, str] = {}
    if DOCS_EN_PATH and os.path.exists(DOCS_EN_PATH):
        with open(DOCS_EN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                docid = str(j.get("docid"))
                content = (j.get("content") or "").strip()
                if docid and content:
                    docs_en_map[docid] = content

    # docid 합집합
    all_docids = sorted(set(docs_ko_map.keys()) | set(docs_en_map.keys()))
    docs_joined: List[Dict[str, Any]] = []
    for docid in all_docids:
        cko = docs_ko_map.get(docid, "")
        cen = docs_en_map.get(docid, "")
        if not cko and not cen:
            continue
        docs_joined.append({"docid": docid, "content_ko": cko, "content_en": cen})

    print(f"[DATA] docs_joined: {len(docs_joined)}")

    # 3) 임베딩 생성 (KO/EN 각각)
    ko_texts = [(d.get("content_ko") or "") for d in docs_joined]
    en_texts = [(d.get("content_en") or "") for d in docs_joined]

    # 빈 텍스트는 임베딩 계산에 넣으면 낭비라서 처리
    # - ko/en 각각 임베딩 리스트는 doc 순서와 맞춰야 하므로, 빈 경우 0벡터(또는 아무 벡터) 대신 None 넣고 ES에는 넣지 않는 방식이 이상적
    # - ES dense_vector는 None 불가라서, 빈 경우 " "로 임베딩 만들어 넣되, 어차피 BM25/knn에서 크게 안 올라오게 두는 형태로 간단 처리
    ko_texts_safe = [t if t.strip() else " " for t in ko_texts]
    en_texts_safe = [t if t.strip() else " " for t in en_texts]

    emb_ko = get_embeddings_in_batches_texts(ko_texts_safe, batch_size=64)
    emb_en = get_embeddings_in_batches_texts(en_texts_safe, batch_size=64)

    # 4) 색인 문서 구성
    index_docs: List[Dict[str, Any]] = []
    for d, vko, ven in zip(docs_joined, emb_ko, emb_en):
        dd = {
            "docid": d["docid"],
            "content_ko": d.get("content_ko", ""),
            "content_en": d.get("content_en", ""),
            "embeddings_ko": vko,
            "embeddings_en": ven,
        }
        index_docs.append(dd)

    # 5) bulk indexing
    ret = bulk_add(ES_INDEX, index_docs, chunk_size=500)
    print("[INDEX] bulk result:", ret)

    # 6) 평가 실행 (KO/EN eval 앙상블)
    eval_rag_bilingual(EVAL_KO_PATH, EVAL_EN_PATH, OUT_JSONL_PATH)
    print(f"[DONE] wrote: {OUT_JSONL_PATH}")