import asyncio, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag.retriever import hybrid_retriever
from src.db.milvus_client import milvus_client

TEST_QUERIES = [
    ("头痛发烧喉咙痛", ["感冒", "流感", "上呼吸道感染"], "症状查询"),
    ("布洛芬有什么副作用", ["布洛芬", "非甾体抗炎药", "止痛药"], "药品查询"),
    ("胸痛呼吸困难怎么办", ["心肌梗死", "急救", "胸痛", "冠心病"], "急诊症状"),
    ("经常反酸烧心是什么病", ["胃食管反流", "烧心", "消化系统疾病"], "症状查询"),
    ("皮肤上起红疹很痒", ["湿疹", "过敏", "皮肤科", "荨麻疹"], "症状查询"),
    ("高血压怎么治疗", ["高血压", "心血管疾病", "慢性病"], "疾病查询"),
    ("突然头晕耳鸣站不稳", ["脑卒中", "头晕", "贫血"], "症状查询"),
    ("吃奥美拉唑要注意什么", ["奥美拉唑", "胃药", "质子泵抑制剂"], "药品查询"),
]

async def main():
    milvus_client.connect()
    for i, (query, expect, cat) in enumerate(TEST_QUERIES):
        results = await hybrid_retriever.retrieve(query, top_k=5)
        print(f"{'='*80}")
        print(f"[{i+1}/8] [{cat}] 输入: {query}")
        print(f"      期望标签: {expect}")
        print(f"{'='*80}")
        for j, r in enumerate(results):
            score = r.get("rrf_score", r.get("score", "N/A"))
            tags = r.get("disease_tag", [])
            print(f"  --- Result #{j+1} ---")
            print(f"  类型: {r.get('source_type','?')}  |  标签: {tags}")
            print(f"  分数: {score}")
            print(f"  内容: {r.get('content','')}")
            print()
        print()

if __name__ == "__main__":
    asyncio.run(main())