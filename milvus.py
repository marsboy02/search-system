import json

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import numpy as np

i = 0


# milvus 컬렉션 생성 및 인덱스 생성
def create_milvus_collection_and_index(host="localhost", port="19530"):
    connections.connect(alias="default", host=host, port=port)

    # 필드 정의
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),  # 기본 키 필드
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=512),  # 문자열 필드
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768)  # 벡터 필드
    ]

    # 컬렉션 스키마 정의
    schema = CollectionSchema(fields, description="search-system recommendation collection")

    # milvus 인스턴스 생성
    milvus = Collection(name="search", schema=schema)
    print("Collection created:", milvus.name)

    print("Start Creating index IVF_FLAT")
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }

    milvus.create_index("embeddings", index)
    print("Index created")

    return milvus


def insert_embedding(milvus, sentence, embedding):
    global i
    entity = [
        {"id": i, "sentence": sentence, "embeddings": embedding.tolist()[0]}
    ]
    i += 1
    milvus.insert(entity)
    milvus.flush()
    print("Data inserted successfully")
    return None  # None 반환


def vector_search(milvus, embedding):
    query_embedding = embedding.numpy().astype(np.float32).tolist()
    print(query_embedding)

    search_params = {
        "metric_type": "COSINE",
        "params": {"nprobe": 10},
    }

    response = milvus.search(
        data=query_embedding,
        anns_field="embeddings",
        param=search_params,
        limit=5,
        output_fields=["sentence"]
    )

    return response

