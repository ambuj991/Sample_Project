import os
from openai import OpenAI
import numpy as np
import tiktoken
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

class CrustDataRAG:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", cache_dir: str = "cache"):
        """Initialize the RAG system with OpenAI credentials and caching."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.documents = []
        self.embeddings_cache = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap for better context preservation."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk)
            chunks.append(chunk_text)
            
        return chunks

    def load_documents(self, file_paths: List[str], force_reload: bool = False):
        """Load documents with caching support."""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        if not force_reload and cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.documents = cached_data['documents']
                self.embeddings_cache = cached_data['embeddings']
                return

        self.documents = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = self.chunk_text(text)
                self.documents.extend(chunks)
        
        self._create_embeddings()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings_cache
            }, f)

    def _create_embeddings(self):
        """Create embeddings in batches for efficiency."""
        self.embeddings_cache = []
        batch_size = 100
        
        for i in range(0, len(self.documents), batch_size):
            batch = self.documents[i:i + batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            self.embeddings_cache.extend([data.embedding for data in response.data])

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Enhanced semantic search with better relevance scoring."""
        query_response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings_cache
        )[0]
        
        threshold = 0.3
        relevant_indices = np.where(similarities > threshold)[0]
        top_indices = relevant_indices[np.argsort(similarities[relevant_indices])[-top_k:]][::-1]
        
        return [(self.documents[i], similarities[i]) for i in top_indices]

    def answer_query(self, query: str, chat_history: List[dict] = None) -> str:
        """Generate response with chat history context."""
        relevant_docs = self.search(query)
        context = "\n".join([f"Document {i+1}:\n{doc[0]}\n" 
                           for i, doc in enumerate(relevant_docs)])
        
        messages = [
            {"role": "system", "content": """You are CrustData's API support specialist. 
             Provide accurate, technical information about CrustData's APIs based on the context.
             Keep responses concise and technical. If information is not in the context, 
             clearly state that and suggest contacting CrustData support."""}
        ]
        
        if chat_history:
            messages.extend(chat_history[-5:])
            
        messages.append({
            "role": "user",
            "content": f"Question about CrustData API: {query}\n\nRelevant documentation:\n{context}"
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content