import json

import chromadb
import jieba
import pandas as pd
from chromadb import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from vanna.base import VannaBase

from vanna.chromadb import ChromaDB_VectorStore
from vanna.chromadb.chromadb_vector import default_ef
from vanna.utils import deterministic_uuid

# 简单的中文停用词表，可以根据需要扩展
stop_words = ["的", "了", "是", "在", "和", "也", "就", "都", "很"]


# 获取查询语句中的中文关键字
def extract_keywords(text):
    words = jieba.lcut(text)
    return [word for word in words if word not in stop_words]


class My_ChromaDB_VectorStore(ChromaDB_VectorStore):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config is None:
            config = {}

        path = config.get("path", ".")
        self.embedding_function = config.get("embedding_function", default_ef)
        curr_client = config.get("client", "persistent")
        vectorDbName = config.get("vectorDbName", "test")
        collection_metadata = config.get("collection_metadata", None)
        self.n_results_sql = config.get("n_results_sql", config.get("n_results", 10))
        self.n_results_documentation = config.get("n_results_documentation", config.get("n_results", 10))
        self.n_results_ddl = config.get("n_results_ddl", config.get("n_results", 10))

        if curr_client == "persistent":
            self.chroma_client = chromadb.PersistentClient(
                path=path, settings=Settings(anonymized_telemetry=False)
            )
        elif curr_client == "in-memory":
            self.chroma_client = chromadb.EphemeralClient(
                settings=Settings(anonymized_telemetry=False)
            )
        elif isinstance(curr_client, chromadb.api.client.Client):
            # allow providing client directly
            self.chroma_client = curr_client
        else:
            raise ValueError(f"Unsupported client was set in config: {curr_client}")

        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name=vectorDbName + "-documentation",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name=vectorDbName + "-ddl",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name=vectorDbName + "-sql",
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

    # 移除训练数据
    def remove_training_data(self, id: str, **kwargs) -> bool:
        if id.endswith("-sql"):
            self.sql_collection.delete(ids=id);
            return True
        elif id.endswith("-ddl"):
            self.ddl_collection.delete(ids=id)
            return True
        elif id.endswith("-doc"):
            self.documentation_collection.delete(ids=id)
            return True
        else:
            return False

    def get_training_data(self, **kwargs) -> pd.DataFrame:
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["sql"] for doc in documents],
                }
            )

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {
                    "id": ids,
                    "question": [doc["question"] for doc in documents],
                    "content": [doc["ddl"] for doc in documents],
                }
            )

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {
                    "id": ids,
                    "question": [None for doc in documents],
                    "content": [doc for doc in documents],
                }
            )

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    # 添加问题与SQL
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            },
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=id,
        )

        return id

    # 添加DDL语句
    def add_ddl(self, question: str, ddl: str, **kwargs) -> str:
        question_ddl_json = json.dumps(
            {
                "question": question,
                "ddl": ddl,
            },
            ensure_ascii=False,
        )
        id = deterministic_uuid(question_ddl_json) + "-ddl"
        self.ddl_collection.add(
            documents=question_ddl_json,
            embeddings=self.generate_embedding(question_ddl_json),
            ids=id,
        )

        return id

    # 获取相似问题的 SQL
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        # 如果 question 为空，则直接返回所有结果
        if not question:
            return My_ChromaDB_VectorStore._extract_documents(
                self.sql_collection.query(
                    query_texts=[question],
                    n_results=10
                )
            )

        # 初步获取查询结果
        documents = My_ChromaDB_VectorStore._extract_documents(
            self.sql_collection.query(
                query_texts=[question],
                n_results=99999
            )
        )

        if not documents:
            return []

        # 使用 TF-IDF 计算相似度
        vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, stop_words=stop_words)
        # 提取 'question' 字段中的文本
        doc_texts = [doc['question'] for doc in documents if 'question' in doc]
        doc_texts.append(question)  # 将查询问题加入文本列表

        # 转换为 TF-IDF 矩阵
        tfidf_matrix = vectorizer.fit_transform(doc_texts)

        # 计算查询问题与文档之间的余弦相似度
        cosine_similarities = np.array(tfidf_matrix[-1].dot(tfidf_matrix[:-1].T).toarray()[0])

        # 找到最高相似度的文档 只返回最高的第一个
        if cosine_similarities.size > 0:
            max_index = np.argmax(cosine_similarities)
            best_document = documents[max_index]
            return [best_document]  # 返回最佳文档
        else:
            return []
