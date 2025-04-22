# hmafqa/retrieval/table_retriever.py
import os
import logging
from typing import Dict, Any, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TableRetriever:
    """
    Retriever for finding relevant tables based on embeddings similarity.
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the table retriever.
        
        Args:
            index_path: Path to the table index
            embedding_model: Model to use for embeddings
        """
        self.index_path = index_path
        
        # Load embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # Load index or initialize empty
        self.table_embeddings = {}
        self.tables = {}
        
        self._load_index()
    
    def _load_index(self):
        """Load the table index from disk."""
        try:
            if os.path.exists(os.path.join(self.index_path, "embeddings.npy")):
                # Load embeddings
                self.table_embeddings = np.load(
                    os.path.join(self.index_path, "embeddings.npy"),
                    allow_pickle=True
                ).item()
                
                # Load tables
                import pickle
                with open(os.path.join(self.index_path, "tables.pkl"), 'rb') as f:
                    self.tables = pickle.load(f)
                
                logger.info(f"Loaded {len(self.tables)} tables from index")
            else:
                logger.warning(f"No index found at {self.index_path}, initializing empty index")
                self.table_embeddings = {}
                self.tables = {}
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            self.table_embeddings = {}
            self.tables = {}
    
    def add_table(
        self,
        table_id: str,
        title: str,
        description: str = "",
        html: Optional[str] = None,
        csv: Optional[str] = None,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a table to the index.
        
        Args:
            table_id: Table ID
            title: Table title
            description: Table description
            html: HTML representation of the table
            csv: CSV representation of the table
            text: Text representation of the table
            metadata: Optional metadata
        """
        # Prepare table for indexing
        table_content = f"{title}\n{description}\n"
        if text:
            table_content += text
        
        # Compute embedding
        embedding = self.model.encode(table_content)
        
        # Store embedding and table
        self.table_embeddings[table_id] = embedding
        self.tables[table_id] = {
            "id": table_id,
            "title": title,
            "description": description,
            **({"html": html} if html else {}),
            **({"csv": csv} if csv else {}),
            **({"text": text} if text else {}),
            **(metadata or {})
        }
    
    def save_index(self):
        """Save the index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save embeddings
        np.save(
            os.path.join(self.index_path, "embeddings.npy"),
            self.table_embeddings
        )
        
        # Save tables
        import pickle
        with open(os.path.join(self.index_path, "tables.pkl"), 'wb') as f:
            pickle.dump(self.tables, f)
        
        logger.info(f"Saved {len(self.tables)} tables to index")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant tables for a query.
        
        Args:
            query: The query string
            top_k: Number of tables to retrieve
            
        Returns:
            List of table dictionaries
        """
        if not self.tables:
            logger.warning("No tables in index")
            return []
        
        # Compute query embedding
        query_embedding = self.model.encode(query)
        
        # Compute similarities
        similarities = {}
        for table_id, table_embedding in self.table_embeddings.items():
            # Compute cosine similarity
            similarity = cosine_similarity(
                [query_embedding],
                [table_embedding]
            )[0][0]
            similarities[table_id] = similarity
        
        # Sort by similarity
        sorted_tables = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k tables
        results = []
        for table_id, similarity in sorted_tables[:top_k]:
            table = self.tables[table_id].copy()
            table["similarity"] = float(similarity)
            results.append(table)
        
        return results