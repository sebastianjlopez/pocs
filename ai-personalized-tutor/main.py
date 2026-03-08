import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document

# 1. Setup sample "curriculum" data
curriculum_data = [
    "Photosynthesis is the process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",
    "The chemical equation is 6CO2 + 6H2O -> C6H12O6 + 6O2."
]

# 2. Initialize Vector Store
embeddings = OpenAIEmbeddings()
docs = [Document(page_content=text) for text in curriculum_data]
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. Setup LLM-driven Tutor
llm = ChatOpenAI(model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

# 4. Demonstrate personalized learning
query = "Explain photosynthesis like I am a 5 year old."
response = qa_chain.invoke(query)

print(f"Student Query: {query}")
print(f"Tutor Response: {response['result']}")