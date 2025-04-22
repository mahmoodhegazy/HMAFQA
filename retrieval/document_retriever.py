# hmafqa/retrieval/document_retriever.py
import os
import logging
from typing import Dict, Any, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Retriever for finding relevant documents based on embeddings similarity.
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the document retriever.
        
        Args:
            index_path: Path to the document index
            embedding_model: Model to use for embeddings
        """
        self.index_path = index_path
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Load index or initialize empty
        self.doc_embeddings = {}
        self.documents = {}
        
        self._load_index()
    
    def _load_index(self):
        """Load the document index from disk."""
        try:
            if os.path.exists(os.path.join(self.index_path, "embeddings.npy")):
                # Load embeddings
                self.doc_embeddings = np.load(
                    os.path.join(self.index_path, "embeddings.npy"),
                    allow_pickle=True
                ).item()
                
                # Load documents
                import pickle
                with open(os.path.join(self.index_path, "documents.pkl"), 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded {len(self.documents)} documents from index")
            else:
                logger.warning(f"No index found at {self.index_path}, initializing empty index")
                self.doc_embeddings = {}
                self.documents = {}
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.doc_embeddings = {}
            self.documents = {}
    
    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a document to the index.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Optional metadata
        """
        # Compute embedding
        embedding = self.model.encode(text)
        
        # Store embedding and document
        self.doc_embeddings[doc_id] = embedding
        self.documents[doc_id] = {
            "id": doc_id,
            "text": text,
            **(metadata or {})
        }
    
    def save_index(self):
        """Save the index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save embeddings
        np.save(
            os.path.join(self.index_path, "embeddings.npy"),
            self.doc_embeddings
        )
        
        # Save documents
        import pickle
        with open(os.path.join(self.index_path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved {len(self.documents)} documents to index")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of document dictionaries
        """
        if not self.documents:
            logger.warning("No documents in index")
            return []
        
        # Compute query embedding
        query_embedding = self.model.encode(query)
        
        # Compute similarities
        similarities = {}
        for doc_id, doc_embedding in self.doc_embeddings.items():
            # Compute cosine similarity
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities[doc_id] = similarity
        
        # Sort by similarity
        sorted_docs = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k documents
        results = []
        for doc_id, similarity in sorted_docs[:top_k]:
            doc = self.documents[doc_id].copy()
            doc["similarity"] = float(similarity)
            results.append(doc)
        
        return results