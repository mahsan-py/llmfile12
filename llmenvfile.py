import langchain
import openai
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import openai
import os
def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents
doc=read_doc('C:/Users/HC/Documents/vector db in AI/doc')
len(doc)
def chunk_data(docs,chunk_size=500,chunk_overlap=10):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    
    return docs

documents=chunk_data(doc)

embeddings=OpenAIEmbeddings(api_key="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")
embeddings
vectors=embeddings.embed_query("Notice period")
len(vectors)

import pinecone
pinecone.init(
    api_key='47af1a3b-0326-461a-bc73-dac25e428644',  # find at app.pinecone.io
    environment="gcp-starter",
)
index_name="langchainvectordb"
index = Pinecone.from_documents(doc, embeddings, index_name=index_name)
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
llm=OpenAI(model_name="gpt-3.5-turbo-0613",temperature=0.1)
chain=load_qa_chain(llm,chain_type="stuff")
def reterieve_query(query,k=1):
    matching_result=index.similarity_search(query,k=k)
    return matching_result
def retrieve_answers(query):
    doc_search=reterieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response
our_query = input("Enter the Question Here : ")

answer = retrieve_answers(our_query)

print("...................")
print("...")
print(our_query)
answer
    


