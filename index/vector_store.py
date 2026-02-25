import numpy as np
# NumPy 2.x compatibility monkeypatch
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool

import chromadb

from chromadb.config import Settings
import uuid

class VectorStore:
    def __init__(self, collection_name="video_rag"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_frames(self, video_id, processed_data):
        """
        Adds frame embeddings and metadata to ChromaDB.
        """
        ids = [str(uuid.uuid4()) for _ in processed_data]
        embeddings = [data['embedding'] for data in processed_data]
        metadatas = [{
            "video_id": video_id,
            "timestamp": data['timestamp'],
            "frame_path": data['frame_path']
        } for data in processed_data]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Added {len(ids)} frames to vector store.")

    def search(self, query_embedding, n_results=5):
        """
        Performs similarity search.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def get_temporal_context(self, best_match_timestamp, video_id, window_seconds=5):
        """
        Fetches context frames around a specific timestamp.
        """
        # Note: ChromaDB doesn't support complex range queries easily in some versions
        # A simple approach for a hackathon: fetch all for video and filter locally or use metadata filters if supported
        results = self.collection.get(
            where={"video_id": video_id}
        )
        
        context_frames = []
        for i, meta in enumerate(results['metadatas']):
            ts = meta['timestamp']
            if abs(ts - best_match_timestamp) <= window_seconds:
                context_frames.append(meta)
        
        # Sort by timestamp
        context_frames.sort(key=lambda x: x['timestamp'])
        return context_frames

if __name__ == "__main__":
    # Test stub
    pass
