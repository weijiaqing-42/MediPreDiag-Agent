# -*- coding: utf-8 -*-
import sys, os, inspect, json, yaml

PASS = FAIL = SKIP = 0
def ck(label, ok, detail=''):
    global PASS, FAIL
    if ok: PASS += 1; print(f'  [PASS] {label}')
    else: FAIL += 1; print(f'  [FAIL] {label}  -- {detail}')
def sk(label):
    global SKIP; SKIP += 1; print(f'  [SKIP] {label}')

print('='*60)
print('MediPreDiag-Agent 验证结果')
print('='*60)

# 1 files
print()
print('[1] 文件完整性')
base = os.path.dirname(os.path.dirname(__file__))
files = [
    'src/config.py','src/main.py','src/agents/state.py','src/agents/graph.py',
    'src/agents/symptom_agent.py','src/agents/severity_evaluator.py','src/agents/medical_advisor.py',
    'src/agents/location_agent.py','src/agents/drug_qa_agent.py','src/agents/response_synthesizer.py',
    'src/agents/memory_nodes.py','src/intent/classifier.py','src/handlers/interrupt.py',
    'src/memory/short_term.py','src/memory/long_term.py','src/rag/retriever.py',
    'src/tools/amap_poi.py','src/tools/drug_info.py','src/db/mysql.py','src/db/redis_client.py',
    'src/db/milvus_client.py','src/routes/chat.py','src/routes/ws.py','src/schemas/models.py',
    'docker-compose.yml','pyproject.toml','.env','scripts/init.sql','static/index.html',
    'data/medical_knowledge.json','scripts/ingest_knowledge.py','scripts/verify_rag.py',
]
for f in files:
    ck(f'文件: {f}', os.path.isfile(os.path.join(base, f)))

# 2 config
print()
print('[2] 配置')
from src.config import settings
ck('DASHSCOPE_API_KEY', bool(settings.dashscope_api_key))
ck('DASHSCOPE_BASE_URL', 'dashscope.aliyuncs.com' in settings.dashscope_base_url)
ck(f'LLM_MODEL = {settings.llm_model}', bool(settings.llm_model))
ck(f'LLM_VISION_MODEL = {settings.llm_vision_model}', bool(settings.llm_vision_model))
ck(f'LLM_FAST_MODEL = {settings.llm_fast_model}', bool(settings.llm_fast_model))
ck(f'EMBEDDING_MODEL = {settings.embedding_model}', bool(settings.embedding_model))
ck('AMAP_API_KEY', bool(settings.amap_api_key))
ck('MySQL config', all([settings.mysql_host, settings.mysql_user, settings.mysql_password]))
ck('Redis config', settings.redis_host and settings.redis_port > 0)
ck('Milvus config', settings.milvus_host and settings.milvus_port > 0)
ck(f'Session TTL = {settings.session_ttl}s', settings.session_ttl == 1800)
ck(f'Max Retry = {settings.max_retry}', settings.max_retry == 3)
ck(f'Node Timeout = {settings.node_timeout}s', settings.node_timeout == 15)

# 3 state
print()
print('[3] MediState')
from src.agents.state import MediState
req = ['user_id','session_id','user_message','image_url','user_location',
       'intent','intent_confidence','interrupt_flag','retry_count','timeout_flag',
       'rollback_target','extracted_symptoms','possible_diseases','severity_level',
       'rag_context','medical_advice','nearby_places','drug_info','final_response',
       'short_term_history','long_term_summary']
for f in req:
    ck(f'字段: {f}', f in MediState.__annotations__)

# 4 graph
print()
print('[4] LangGraph')
from src.agents.graph import graph, route_by_intent, route_after_severity
nodes = list(graph.nodes.keys())
expected = ['intent_classifier','memory_loader','symptom_analysis','rag_retriever',
            'severity_evaluator','medical_advisor','location_recommender','drug_qa_agent',
            'response_synthesizer','memory_updater','interrupt_handler']
ck(f'节点数: {len(nodes)} (预期 11)', len(nodes) >= 11)
for n in expected:
    ck(f'节点: {n}', n in nodes)

base_state = {
    'user_id':'t','session_id':'t','user_message':'t','image_url':None,'user_location':None,
    'intent':'unknown','intent_confidence':0.0,'interrupt_flag':False,'retry_count':0,
    'timeout_flag':False,'rollback_target':None,'extracted_symptoms':[],'possible_diseases':[],
    'severity_level':'unknown','rag_context':'','medical_advice':'','nearby_places':[],
    'drug_info':'','final_response':'','short_term_history':[],'long_term_summary':'',
}

# 5 routing
print()
print('[5] 条件边路由')
routes_map = {'symptom_query':'symptom_analysis','drug_query':'drug_qa_agent',
              'location_search':'location_recommender','emergency':'severity_evaluator',
              'chitchat':'response_synthesizer','unknown':'response_synthesizer'}
for intent, exp in routes_map.items():
    s = dict(base_state); s['intent'] = intent
    r = route_by_intent(s)
    ck(f'{intent} -> {exp}', r == exp, f'got: {r}')

s = dict(base_state); s['interrupt_flag'] = True
ck('interrupt_flag -> interrupt_handler', route_by_intent(s) == 'interrupt_handler')
s = dict(base_state); s['timeout_flag'] = True
ck('timeout_flag -> interrupt_handler', route_by_intent(s) == 'interrupt_handler')

for sev, exp in [('mild','medical_advisor'),('severe','location_recommender')]:
    s = dict(base_state); s['severity_level'] = sev
    r = route_after_severity(s)
    ck(f'{sev} -> {exp}', r == exp, f'got: {r}')

# 6 intent
print()
print('[6] 意图分类器逻辑')
from src.intent.classifier import _extract_json, _validate_result, INTENT_LABELS
for i in ['symptom_query','drug_query','location_search','emergency','chitchat','unknown']:
    ck(f'标签: {i}', i in INTENT_LABELS)

tests = [
    ('{"intent":"symptom_query","confidence":0.95}', 'symptom_query'),
    ('```json\n{"intent":"drug_query","confidence":0.8}\n```', 'drug_query'),
]
for raw, exp in tests:
    r = _extract_json(raw)
    ck(f'JSON提取: {raw[:25]}...', r is not None and r.get('intent') == exp)

ck('非法intent->unknown', _validate_result({'intent':'bad','confidence':0.5})['intent']=='unknown')
ck('confidence钳制', _validate_result({'intent':'chitchat','confidence':1.5})['confidence']==1.0)
ck('空dict默认填充', _validate_result({})['intent']=='unknown')

# 7 interrupt
print()
print('[7] 中断处理')
from src.handlers.interrupt import InterruptHandler, interrupt_handler
cancels = ['取消','重新开始','算了','不用了','退出','停止']
ck('取消关键词完整', all(kw in InterruptHandler.CANCEL_KEYWORDS for kw in cancels))
ck('max_retry=3', interrupt_handler.max_retry == 3)
ck('timeout=15s', interrupt_handler.timeout_threshold == 15)
for node in ['intent_classifier','symptom_extractor','rag_retriever','location_recommender','severity_evaluator']:
    resp = interrupt_handler.build_fallback_response(node)
    ck(f'回退响应 {node}', len(resp) > 10)

# 8 AMAP
print()
print('[8] 高德POI策略')
from src.tools.amap_poi import amap_tool
for sev, rad, types in [('mild',1000,'010901'),('moderate',3000,'090100'),('severe',5000,'090101')]:
    strat = amap_tool.STRATEGY.get(sev,{})
    ck(f'{sev}: radius={rad}', strat.get('radius')==rad)
    ck(f'{sev}: types含{types}', types in strat.get('types',''))

# 9 agents
print()
print('[9] Agent节点')
from src.agents.symptom_agent import symptom_analysis_entry
from src.agents.severity_evaluator import severity_evaluator_node
from src.agents.medical_advisor import medical_advisor_node
from src.agents.location_agent import location_recommender_node
from src.agents.drug_qa_agent import drug_qa_agent_node
from src.agents.response_synthesizer import response_synthesizer_node
for name, fn in [('symptom_analysis_entry',symptom_analysis_entry),
                 ('severity_evaluator_node',severity_evaluator_node),
                 ('medical_advisor_node',medical_advisor_node),
                 ('location_recommender_node',location_recommender_node),
                 ('drug_qa_agent_node',drug_qa_agent_node),
                 ('response_synthesizer_node',response_synthesizer_node)]:
    ck(name, callable(fn))

# 10 memory
print()
print('[10] 记忆系统')
from src.memory.short_term import short_term_memory
from src.memory.long_term import long_term_memory
ck('short_term_memory单例', short_term_memory is not None)
ck('long_term_memory单例', long_term_memory is not None)
ck('generate_summary', hasattr(long_term_memory, 'generate_summary'))
ck('save_summary', hasattr(long_term_memory, 'save_summary'))
ck('retrieve_summaries', hasattr(long_term_memory, 'retrieve_summaries'))

# 11 RAG
print()
print('[11] RAG检索结构')
from src.rag.retriever import HybridRetriever, BM25Searcher, Reranker, hybrid_retriever
ck('BM25 uses jieba', 'jieba' in inspect.getsource(BM25Searcher))
ck('_rrf_fusion exists', '_rrf_fusion' in inspect.getsource(HybridRetriever))
ck('Reranker.rerank exists', 'async def rerank' in inspect.getsource(Reranker))
ck('Reranker uses LLM', 'chat.completions.create' in inspect.getsource(Reranker))

# 12 FastAPI
print()
print('[12] FastAPI网关')
from src.main import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
ck('POST /api/v1/chat', '/api/v1/chat' in routes)
ck('GET /api/v1/session/{session_id}', '/api/v1/session/{session_id}' in routes)
ck('DELETE /api/v1/session/{session_id}', '/api/v1/session/{session_id}' in routes)
ck('WS /ws/chat', '/ws/chat' in routes)
ck('GET /health', '/health' in routes)
ck('GET /', '/' in routes)

# 13 docker compose
print()
print('[13] Docker Compose')
with open(os.path.join(base, 'docker-compose.yml'), 'r', encoding='utf-8') as f:
    dc = yaml.safe_load(f)
svcs = dc.get('services', {})
for svc in ['mysql','redis','etcd','minio','milvus']:
    ck(f'服务: {svc}', svc in svcs)
ck('MySQL healthcheck', 'healthcheck' in svcs.get('mysql',{}))
ck('Redis healthcheck', 'healthcheck' in svcs.get('redis',{}))
ck('Milvus healthcheck', 'healthcheck' in svcs.get('milvus',{}))

# 14 MySQL schema
print()
print('[14] MySQL Schema')
from src.db.mysql import init_mysql_schema
src_sql = inspect.getsource(init_mysql_schema)
tables = {
    'users': ['id','phone','nickname','gender','birth_year','allergy_info','chronic_conditions','created_at'],
    'sessions': ['id','user_id','start_time','end_time','intent_path','is_emergency','status'],
    'diagnoses': ['id','session_id','extracted_symptoms','possible_diseases','severity_level','medical_advice'],
    'drug_queries': ['id','user_id','drug_name','query_content','response','created_at'],
}
for table, cols in tables.items():
    for col in cols:
        ck(f'{table}.{col}', col in src_sql)

# 15 libraries
print()
print('[15] 核心库导入')
libs = {'langgraph':'Agent编排','langchain':'LangChain','pymilvus':'Milvus','pymysql':'MySQL',
        'redis':'Redis','sqlalchemy':'ORM','fastapi':'FastAPI','jieba':'分词','httpx':'HTTP','yaml':'YAML'}
for lib, desc in libs.items():
    try:
        __import__(lib); ck(f'{lib} ({desc})', True)
    except ImportError as e:
        ck(f'{lib} ({desc})', False, str(e)[:60])

# 16 knowledge data
print()
print('[16] 知识库数据')
kf = os.path.join(base, 'data', 'medical_knowledge.json')
ck('medical_knowledge.json exists', os.path.isfile(kf))
if os.path.isfile(kf):
    with open(kf, 'r', encoding='utf-8') as f:
        kd = json.load(f)
    ck(f'文档数: {len(kd)} (预期 50)', len(kd) >= 50)
    types = {}
    for d in kd:
        t = d.get('source_type','?')
        types[t] = types.get(t,0)+1
    for t,cnt in sorted(types.items()):
        print(f'          {t}: {cnt} 篇')

# 17 Milvus connection
print()
print('[17] Milvus数据状态')
try:
    from src.db.milvus_client import milvus_client as mc
    mc.connect()
    coll = mc.get_collection()
    n = coll.num_entities
    if n > 0:
        ck(f'已灌库: {n} entities', True)
    else:
        ck(f'Collection为空: 0 entities', False, '请运行 scripts/ingest_knowledge.py')
except Exception as e:
    msg = str(e)[:80]
    print(f'  [SKIP] Milvus未连接: {msg}')
    print(f'         请先 docker compose up -d 启动基础设施')

# summary
print()
print('='*60)
print(f'结果: {PASS} PASS / {FAIL} FAIL / {SKIP} SKIP')
print('='*60)