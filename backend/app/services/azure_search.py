import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, VectorSearch, 
    VectorSearchAlgorithmConfiguration
)
from typing import List, Dict, Any
import json

from config import settings

class AzureSearchService:
    """Service for Azure Cognitive Search operations"""
    
    def __init__(self):
        self.endpoint = settings.AZURE_SEARCH_ENDPOINT
        self.key = settings.AZURE_SEARCH_API_KEY
        self.index_name = settings.AZURE_SEARCH_INDEX
        
        # Create search clients
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, 
            credential=AzureKeyCredential(self.key)
        )
        
        self.search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.key)
        )
        
        # Ensure index exists
        self._create_index_if_not_exists()
    
    def _create_index_if_not_exists(self):
        """Create search index if it doesn't exist"""
        # Check if index exists in the list of index names
        if self.index_name in [index.name for index in self.index_client.list_index_names()]:
            return
            
        # Define vector search configuration
        vector_search = VectorSearch(
            algorithms=[
                VectorSearchAlgorithmConfiguration(
                    name="hnsw",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500
                    }
                )
            ]
        )
        
        # Define fields for the index
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=384,
                vector_search_configuration="hnsw"
            ),
            SearchField(name="metadata", type=SearchFieldDataType.String, filterable=True)
        ]
        
        # Create the index
        index = SearchIndex(
            name=self.index_name, 
            fields=fields, 
            vector_search=vector_search
        )
        self.index_client.create_or_update_index(index)
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: List[List[float]], 
        ids: List[str],
        metadatas: List[Dict[str, Any]] = None
    ):
        """Add documents to the vector store"""
        if not documents:
            return
        
        # Prepare documents for indexing
        docs_to_upload = []
        for i, (doc, emb, doc_id) in enumerate(zip(documents, embeddings, ids)):
            metadata = {}
            if metadatas and i < len(metadatas):
                metadata = metadatas[i]
            
            docs_to_upload.append({
                "id": doc_id,
                "content": doc,
                "embedding": emb,
                "metadata": json.dumps(metadata)
            })
        
        # Upload in batches of 1000 (Azure limit)
        batch_size = 1000
        for i in range(0, len(docs_to_upload), batch_size):
            batch = docs_to_upload[i:i+batch_size]
            try:
                self.search_client.upload_documents(batch)
            except Exception as e:
                print(f"Error uploading documents batch {i}-{i+len(batch)}: {e}")
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Query the vector index
        results = self.search_client.search(
            search_text="*",
            vector={"value": query_embedding, "fields": "embedding", "k": k}
        )
        
        # Format results
        documents = []
        for result in results:
            metadata = {}
            if result.get("metadata"):
                try:
                    metadata = json.loads(result["metadata"])
                except:
                    pass
                    
            documents.append({
                "id": result["id"],
                "content": result["content"],
                "metadata": metadata
            })
        
        return documents

# Singleton instance
azure_search_service = AzureSearchService()