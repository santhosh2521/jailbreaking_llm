import requests
import os
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GROQ_API_KEY"]=""
loader = WebBaseLoader("https://cph-sec.gitbook.io/ai-llm-red-team-handbook-and-field-manual/part-v-attacks-and-techniques/chapter_16_jailbreaks_and_bypass_techniques")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,   # Set chunk size to 512 characters
    length_function=len
)
#listToStr = ' '.join([str(element) for element in data]) #todo:remove this not usefull halfway through
docs = text_splitter.split_documents(data)
#print(docs)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)
#print(db)
retriever = db.as_retriever()
llm = ChatGroq(model="qwen/qwen3-32b")
template = """Use the following pieces of context to answer the question at the end and ignore Testing Methodologies from the context.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
print(rag_chain.invoke("What are some jailbreaking techniques and generate some example snippets for those attacks"))