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
        
    def test_api_key(self) -> bool:
        """Test if the API key is valid."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
            return True
        except Exception as e:
            raise Exception(f"API key validation failed: {str(e)}")

    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap for better context preservation."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk)
            chunks.append(chunk_text)
            
        return chunks

    def _create_embeddings(self):
        """Create embeddings in batches for efficiency."""
        self.embeddings_cache = []
        batch_size = 100
        
        try:
            for i in range(0, len(self.documents), batch_size):
                batch = self.documents[i:i + batch_size]
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                self.embeddings_cache.extend([data.embedding for data in response.data])
                
            print(f"Created {len(self.embeddings_cache)} embeddings")
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            raise

    def load_documents(self, file_paths: List[str], force_reload: bool = False):
        """Load documents with caching support."""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        if not force_reload and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.documents = cached_data.get('documents', [])
                    self.embeddings_cache = cached_data.get('embeddings', [])
                    if self.documents and self.embeddings_cache:
                        print(f"Loaded {len(self.documents)} documents from cache")
                        return
            except Exception as e:
                print(f"Cache loading failed: {e}")

        self.documents = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    if text:
                        chunks = self.chunk_text(text)
                        self.documents.extend(chunks)
                        print(f"Loaded {len(chunks)} chunks from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if not self.documents:
            raise ValueError("No documents were successfully loaded")
            
        self._create_embeddings()
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings_cache
                }, f)
            print("Saved to cache successfully")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Enhanced semantic search with better relevance scoring."""
        if not self.documents or not self.embeddings_cache:
            return []

        try:
            # Get query embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = response.data[0].embedding

            # Prepare arrays
            query_vector = np.array(query_embedding).reshape(1, -1)
            doc_vectors = np.array(self.embeddings_cache)
            
            if len(doc_vectors.shape) == 1:
                doc_vectors = doc_vectors.reshape(1, -1)
                
            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Find relevant documents
            threshold = 0.3
            relevant_indices = np.where(similarities > threshold)[0]
            
            if len(relevant_indices) == 0:
                return []
            
            # Get top results
            top_k = min(top_k, len(relevant_indices))
            top_indices = relevant_indices[np.argsort(similarities[relevant_indices])[-top_k:]][::-1]
            
            return [(self.documents[i], float(similarities[i])) for i in top_indices]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def answer_query(self, query: str, chat_history: List[dict] = None) -> str:
        """Generate response with chat history context."""
        try:
            relevant_docs = self.search(query)
            if not relevant_docs:
                # Handle different types of queries
                if any(greeting in query.lower() for greeting in ['hello', 'hi', 'hey', 'greetings']):
                    return """Hello! 👋 I'm Clara, CrustData's AI Assistant. I'm here to help you with:

• Understanding CrustData's APIs and their features
• Implementation guidance and code examples
• Best practices and troubleshooting
• Authentication and rate limiting questions

How can I assist you with CrustData's APIs today?"""
                
                if 'what do you do' in query.lower() or 'who are you' in query.lower():
                    return """I'm Clara, CrustData's AI Assistant! 🤖 I specialize in helping developers like you with CrustData's APIs. I can:

• Guide you through API endpoints and features
• Provide code examples and implementation tips
• Help with authentication and error handling
• Share best practices and optimization techniques

What aspect of CrustData's APIs would you like to learn more about?"""
                
                return """I don't have specific information about that in my current documentation. However, I'd be happy to help you with:

• API authentication and setup
• Available endpoints and their usage
• Rate limits and best practices
• Error handling and troubleshooting

Would you like to learn more about any of these topics? Alternatively, you can check our [official documentation](https://docs.crustdata.com) or contact our support team for more specific assistance."""

            context = "\n".join([f"Document {i+1}:\n{doc[0]}\n" 
                               for i, doc in enumerate(relevant_docs)])
            
            messages = [
                {"role": "system", "content": """You are Clara, CrustData's friendly and knowledgeable AI Assistant. Your role is to help developers understand and use CrustData's APIs effectively.

Key traits and behavior guidelines:
- Be welcoming and professional while maintaining a conversational tone
- When greeting users for the first time, briefly introduce yourself and explain how you can help with CrustData's APIs
- For general questions about what you do, explain that you're an AI assistant specializing in CrustData's API documentation and support
- Provide technically accurate information based on the available documentation
- Use clear explanations with code examples when relevant
- If information isn't in the context but you know it's API-related, suggest possible topics to explore
- For completely off-topic questions, politely redirect to CrustData API topics
- When information is not available in the documentation, suggest checking the official documentation or contacting support

Format your responses with:
- Clear structure using paragraphs and bullet points when appropriate
- Code examples in markdown format
- Highlight important points or warnings
- End with an invitation for follow-up questions when appropriate"""}
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
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error while processing your request. Please try again or contact support if the issue persists."