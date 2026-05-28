"""
RAG Retrieval Verification Script
End-to-end test of the hybrid retrieval pipeline.
Tests: Vector search, BM25, RRF fusion, LLM rerank on real medical queries.
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.retriever import hybrid_retriever
from src.db.milvus_client import milvus_client


TEST_QUERIES = [
    {
        "query": "头痛发烧喉咙痛",
        "expect_tags": ["感冒", "流感", "上呼吸道感染"],
        "category": "症状查询",
    },
    {
        "query": "布洛芬有什么副作用",
        "expect_tags": ["布洛芬", "非甾体抗炎药", "止痛药"],
        "category": "药品查询",
    },
    {
        "query": "胸痛呼吸困难怎么办",
        "expect_tags": ["心肌梗死", "急救", "胸痛", "冠心病"],
        "category": "急诊症状",
    },
    {
        "query": "经常反酸烧心是什么病",
        "expect_tags": ["胃食管反流", "烧心", "消化系统疾病"],
        "category": "症状查询",
    },
    {
        "query": "皮肤上起红疹很痒",
        "expect_tags": ["湿疹", "过敏", "皮肤科", "荨麻疹"],
        "category": "症状查询",
    },
    {
        "query": "高血压怎么治疗",
        "expect_tags": ["高血压", "心血管疾病", "慢性病"],
        "category": "疾病查询",
    },
    {
        "query": "突然头晕耳鸣站不稳",
        "expect_tags": ["脑卒中", "头晕", "贫血"],
        "category": "症状查询",
    },
    {
        "query": "吃奥美拉唑要注意什么",
        "expect_tags": ["奥美拉唑", "胃药", "质子泵抑制剂"],
        "category": "药品查询",
    },
]


def verify_result(query: str, results: list, expect_tags: list, category: str) -> dict:
    if not results:
        return {"status": "FAIL", "reason": "No results returned"}

    top_k = min(5, len(results))
    top_contents = [r.get("content", "") for r in results[:top_k]]
    top_sources = [r.get("source_type", "unknown") for r in results[:top_k]]
    top_tags = [r.get("disease_tag", []) for r in results[:top_k]]

    tag_match = 0
    all_tags = []
    for tags in top_tags:
        all_tags.extend(tags)
    for et in expect_tags:
        if any(et in t for t in all_tags):
            tag_match += 1

    relevance_ok = tag_match >= 1
    source_ok = len(set(top_sources)) >= 1

    if relevance_ok and source_ok:
        return {"status": "PASS", "relevance": f"{tag_match}/{len(expect_tags)} tags matched"}
    elif relevance_ok:
        return {"status": "WARN", "reason": "Tags ok but source diversity low"}
    else:
        return {"status": "WARN", "reason": f"0/{len(expect_tags)} expected tags matched",
                "got_tags": list(set(all_tags))[:10]}


async def main():
    print("=" * 60)
    print("RAG Retrieval Pipeline Verification")
    print("=" * 60)

    print("\n[1] Checking Milvus connection...")
    try:
        milvus_client.connect()
        collection = milvus_client.get_collection()
        num_entities = collection.num_entities
        print(f"  Connected. Collection 'medical_knowledge' has {num_entities} entities.")
        if num_entities == 0:
            print("  [ERROR] Collection is EMPTY. Run 'scripts/ingest_knowledge.py' first.")
            return
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Milvus: {e}")
        print("  Run 'docker compose up -d' first, then run ingest_knowledge.py")
        return

    print("\n[2] Running retrieval tests...\n")
    passed = 0
    failed = 0

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        expect_tags = test["expect_tags"]
        category = test["category"]
        print(f"  Test {i+1}: [{category}] \"{query}\"")

        try:
            results = await hybrid_retriever.retrieve(query, top_k=5)
        except Exception as e:
            print(f"    [FAIL] Retrieval error: {e}")
            failed += 1
            continue

        verdict = verify_result(query, results, expect_tags, category)
        status = verdict["status"]

        if status == "PASS":
            passed += 1
            print(f"    [PASS] {verdict['relevance']}, {len(results)} results")
        else:
            print(f"    [{status}] {verdict.get('reason', '')}")
            if "got_tags" in verdict:
                print(f"           Got tags: {verdict['got_tags']}")
            failed += 1

        if results:
            best = results[0]
            content_preview = best.get("content", "")[:100]
            score = best.get("score", best.get("rrf_score", "N/A"))
            print(f"           Top: [{best.get('source_type','?')}] {content_preview}...")
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(TEST_QUERIES)} passed, {failed} failed/warned")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())