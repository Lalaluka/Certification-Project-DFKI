#!/usr/bin/env python3
"""
Unified RAG utilities for all tasks.
This module provides a consistent interface for Retrieval-Augmented Generation
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional


class UnifiedRAG:
    """
    Unified RAG system that can be used across all tasks.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 knowledge_base_dir: Optional[str] = None):
        """
        Initialize the unified RAG system.
        
        Args:
            model_name: SentenceTransformer model name
            knowledge_base_dir: Path to directory with text files and faq.json
        """
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        
        if knowledge_base_dir and os.path.exists(knowledge_base_dir):
            self._load_from_directory(knowledge_base_dir)
        else:
            print("Warning: No knowledge base provided. Call load_knowledge_base() manually.")
        
    def _load_from_directory(self, knowledge_base_dir: str):
        """Load knowledge base from directory with text files"""
        self.documents = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", ".", "\n\n", "?", "!"],
            keep_separator="end",
            chunk_size=400,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Load text files
        for filename in os.listdir(knowledge_base_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(knowledge_base_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    # Use text splitter to create chunks
                    chunks = text_splitter.split_text(text)
                    self.documents.extend(chunks)
        
        # Load FAQ file
        faq_path = os.path.join(knowledge_base_dir, "faq.json")
        if os.path.exists(faq_path):
            with open(faq_path, "r", encoding="utf-8") as f:
                faqs = json.load(f)
                for entry in faqs:
                    if isinstance(entry, dict) and 'question' in entry and 'answer' in entry:
                        self.documents.append(f"Q: {entry['question']}\nA: {entry['answer']}")
        
        self._build_embeddings()
        print(f"✅ Loaded {len(self.documents)} documents from {knowledge_base_dir}")
    
    def _build_embeddings(self):
        self.embeddings = self.sentence_model.encode(self.documents)
    
    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between query and document embeddings"""
        # Cosine similarity: dot product of normalized vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
        return float(np.dot(query_norm, doc_norm))
    
    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of top documents to retrieve
            min_score: Minimum similarity score (0-1)
        
        Returns:
            List of dicts with 'content', 'score', and 'index' keys
        """        
        # Encode query
        query_embedding = self.sentence_model.encode([query])[0]
        
        # Calculate similarities with all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._calculate_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity (highest first) and take top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Format results
        results = []
        for similarity, idx in similarities[:top_k]:
            if similarity >= min_score:
                results.append({
                    'content': self.documents[idx],
                    'score': similarity,
                    'index': idx
                })
        
        return results
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context string for RAG.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
        
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        # Simply join all results
        context_parts = [result['content'] for result in results]
        
        return "\n\n".join(context_parts)
    
    def get_retrieval_score(self, query: str) -> float:
        """
        Get the maximum retrieval score for agent decision making.
        Used in Task 3 for threshold-based RAG decisions.
        
        Args:
            query: The search query
        
        Returns:
            Maximum similarity score (0-1)
        """
        results = self.retrieve(query, top_k=1)
        return results[0]['score'] if results else 0.0

def create_rag(base_dir: str = "/Users/A79813024/dev/ai_upskilling") -> UnifiedRAG:
    """
    Factory function to create RAG instance
    
    Args:
        base_dir: Base directory path
    
    Returns:
        Configured UnifiedRAG instance
    """
    knowledge_base_dir = os.path.join(base_dir, "task2", "data", "knowledge_base")
    return UnifiedRAG(knowledge_base_dir=knowledge_base_dir)
