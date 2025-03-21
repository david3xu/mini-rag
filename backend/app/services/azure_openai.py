import os
import openai
from typing import List, Dict, Any, Generator

from config import settings

class AzureOpenAIService:
    """Service for generating text with Azure OpenAI"""
    
    def __init__(self):
        # Set up Azure OpenAI client
        openai.api_type = "azure"
        openai.api_base = settings.AZURE_OPENAI_ENDPOINT
        openai.api_version = "2023-05-15"
        openai.api_key = settings.AZURE_OPENAI_API_KEY
        
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT
    
    def generate_text(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """Generate text from Azure OpenAI"""
        # For Azure OpenAI, we use chat completions
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        if not stream:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            # Process streaming response
            for chunk in openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            ):
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    yield chunk.choices[0].delta.content
    
    def format_rag_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Format a prompt for RAG"""
        context = "\n\n".join([
            f"Document {i+1}:\n{doc['content']}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""
        Answer the following query based on the provided documents.
        If you don't know the answer, say you don't know.

        Documents:
        {context}

        Query: {query}

        Answer:
        """
        return prompt

# Singleton instance
azure_openai_service = AzureOpenAIService()