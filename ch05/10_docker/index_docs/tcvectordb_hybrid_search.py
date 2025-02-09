from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from tcvdb_text.encoder import BM25Encoder
from tcvectordb.model.document import AnnSearch, KeywordSearch, WeightedRerank


class TencentVectorDBRetriever(BaseRetriever):
    client: Any
    embeddings: Embeddings
    sparse_encoder: BM25Encoder
    database_name: str
    collection_name: str
    limit: int = 10
    weight: List[float] = [0.5, 0.5]
    field_id: str = "id"
    field_text: str = "text"
    field_vector: str = "vector"
    field_sparse_vector: str = "sparse_vector"

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            hybrid_search_kwargs: dict[str, Any] = {}
    ) -> list[Document]:
        dense_embeddings: List[float] = self.embeddings.embed_query(query)
        sparse_embeddings = self.sparse_encoder.encode_queries(query)
        weight = hybrid_search_kwargs.get("weight", self.weight)
        doc_list = self.client.hybrid_search(
            database_name=self.database_name,
            collection_name=self.collection_name,
            ann=[
                AnnSearch(
                    field_name=self.field_vector,
                    data=dense_embeddings,
                    limit=self.limit
                ),
            ],
            match=[
                KeywordSearch(
                    field_name=self.field_sparse_vector,
                    data=sparse_embeddings,
                    limit=self.limit
                ),
            ],
            rerank=WeightedRerank(
                field_list=[self.field_vector, self.field_sparse_vector],
                weight=weight,
            ),
            retrieve_vector=False,
            limit=self.limit,
        )
        docs = []
        for res in doc_list[0]:
            doc_id = res.pop(self.field_id)
            text = res.pop(self.field_text)
            docs.append(Document(page_content=text, id=doc_id, metadata=res))
        return docs
