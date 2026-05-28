from typing import List, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from src.config import settings

COLLECTION_NAME = "medical_knowledge"
DIMENSION = 1024


class MilvusClient:
    def __init__(self):
        self._connected = False

    def connect(self):
        if not self._connected:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
            )
            self._connected = True

    def disconnect(self):
        if self._connected:
            connections.disconnect("default")
            self._connected = False

    def init_collection(self):
        self.connect()
        if utility.has_collection(COLLECTION_NAME):
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="disease_tag", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        ]
        schema = CollectionSchema(fields, description="Medical Knowledge Base")
        collection = Collection(COLLECTION_NAME, schema)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index("embedding", index_params)
        collection.load()

    def get_collection(self) -> Collection:
        self.connect()
        return Collection(COLLECTION_NAME)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        expr: Optional[str] = None,
    ) -> List[dict]:
        collection = self.get_collection()
        search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content", "source_type", "disease_tag"],
        )
        return [
            {
                "id": hit.id,
                "content": hit.entity.get("content", ""),
                "source_type": hit.entity.get("source_type", ""),
                "disease_tag": hit.entity.get("disease_tag", []),
                "score": hit.score,
            }
            for hit in results[0]
        ]

    def insert(self, contents: List[dict], embeddings: List[List[float]]):
        collection = self.get_collection()
        data = [
            [c["content"] for c in contents],
            [c.get("source_type", "") for c in contents],
            [c.get("disease_tag", []) for c in contents],
            embeddings,
        ]
        collection.insert(data)


milvus_client = MilvusClient()