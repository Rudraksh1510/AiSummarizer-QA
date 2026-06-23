
from pinecone import Pinecone
from dotenv import load_dotenv
from src.pdfsummarizer.logger import logging
from src.pdfsummarizer.utils import extract_text_from_pdf
import os

#langchain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#Loading API keys from .env file
load_dotenv()
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)
api_key = os.getenv("GOOGLE_API_KEY")
logging.info("API keys loaded successfully.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the provided context.

Context:
{context}

Question:
{question}
""")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
vector = embeddings.embed_query(chunks)

index_name='test'
index=pc.Index(index_name)

docsearch = PineconeVectorStore.from_texts(
    texts=chunks,
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(
    search_kwargs={"k": 3}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()   #Runnable passthrough means return the input unchnged
    }
    | prompt
    | llm
    | StrOutputParser() #Make the output printable 
)

while True:
    question = input("Question: ")

    if question.lower() == "exit":
        break

    answer = rag_chain.invoke(question)

    print("\nAnswer:")
    print(answer)
    print()