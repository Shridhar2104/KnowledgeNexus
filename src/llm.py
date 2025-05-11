import os
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

class LLMProcessor:
    """
    A class to handle LLM operations, including prompt formatting and response generation
    based on retrieved documents.
    """
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.2):
        """
        Initialize the LLM processor with a model and API key.
        
        Args:
            model_name: Name of the LLM model (e.g., 'gpt-3.5-turbo')
            api_key: API key for accessing the LLM API
            temperature: Temperature parameter for controlling randomness
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def _initialize_llm(self):
        """
        Initialize the LLM based on the model name.
        
        Returns:
            An initialized LLM
        """
        if 'gpt' in self.model_name.lower():
            # OpenAI models
            try:
                return ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=self.temperature
                )
            except Exception as e:
                print(f"Error initializing OpenAI model: {e}")
                print("Falling back to local model...")
                return self._initialize_local_model()
        else:
            # Other models (like local Hugging Face models)
            return self._initialize_local_model()
            
    def _initialize_local_model(self):
        """Initialize a local Hugging Face model as fallback"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        from langchain_huggingface import HuggingFacePipeline
        
        # Use a smaller model that can run locally
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        print(f"Loading local model: {model_id}...")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=self.temperature,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain HuggingFacePipeline
        return HuggingFacePipeline(pipeline=pipe)
        
    def _create_prompt_template(self):
        """
        Create a prompt template for the LLM.
        
        Returns:
            A PromptTemplate object
        """
        template = """
        You are a helpful assistant that provides accurate information based on the given context.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        INSTRUCTIONS:
        1. Answer the question based only on the information provided in the context.
        2. If the answer is not present in the context, respond with "I don't have enough information to answer this question."
        3. Do not make up or infer information that is not directly stated in the context.
        4. Provide clear and concise answers with relevant details from the context.
        5. Include citations to the relevant documents when appropriate using [doc1], [doc2], etc. notation.
        
        ANSWER:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _format_documents(self, docs: List[Document]) -> str:
        """
        Format the retrieved documents into a context string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', f'Document {i+1}')
            content = doc.page_content.strip()
            formatted_docs.append(f"[doc{i+1}] {content} (Source: {source})")
        
        return "\n\n".join(formatted_docs)
    
    def generate_response(self, question: str, docs: List[Document]) -> str:
        """
        Generate a response to a question using the retrieved documents as context.
        
        Args:
            question: The question to answer
            docs: The retrieved documents to use as context
            
        Returns:
            Generated response
        """
        if not docs:
            return "I don't have any relevant information to answer this question."
        
        formatted_context = self._format_documents(docs)
        
        try:
            response = self.chain.invoke({
                "context": formatted_context,
                "question": question
            })
            
            # Handle different response formats
            if isinstance(response, dict) and 'text' in response:
                return response['text']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
        
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature
        }