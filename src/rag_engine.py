import os
from typing import List, Any
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain Imports
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()


class GraphRAGManager:
    """
    Manages the RAG pipeline: PDF processing, Graph indexing, and Querying.
    Using a class structure makes the code modular and testable (Senior approach).
    """

    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        # Initialize Embeddings (HuggingFace - Free and local)
        # We use 'all-MiniLM-L6-v2' because it's fast and lightweight.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize LLM (Groq - Llama3)
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama3-8b-8192",
            temperature=0,  # 0 for factual consistency
        )

    def process_pdf(self, uploaded_file) -> List[str]:
        """
        Reads a PDF file and splits it into manageable chunks.
        """
        try:
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            # Split text into chunks to avoid hitting context limits
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""],
            )
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def index_documents(self, text_chunks: List[str], index_name: str = "vector_index"):
        """
        Stores the text chunks into Neo4j as a Vector Store.
        This creates a graph structure where nodes contain text and embeddings.
        """
        if not text_chunks:
            return None

        # Neo4jVector automatically creates the graph nodes and vector index
        vector_store = Neo4jVector.from_texts(
            texts=text_chunks,
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            index_name=index_name,
        )
        return vector_store

    def get_qa_chain(self):
        """
        Creates the RetrievalQA chain combining the Vector Store and the LLM.
        """
        # Re-connect to the existing vector index in Neo4j
        vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=self.neo4j_uri,
            username=self.neo4j_user,
            password=self.neo4j_password,
            index_name="vector_index",
        )

        # Create a retriever object
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Custom Prompt to ensure the bot acts professionally
        prompt_template = """
        You are a senior technical assistant. Use the following context to answer the question.
        If you don't know the answer based on the context, say so. Do not make things up.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Build the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
        )
        return qa_chain
