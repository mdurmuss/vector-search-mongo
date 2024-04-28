#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Mustafa Durmuş [mustafa.durmus@albert.health]

# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()


def vectorize(file_path: str) -> None:
    """
    This function vectorizes the documents in the little prince book and saves them into MongoDB Atlas.
    """
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    username = os.getenv("MONGO_DB_ATLAS_USER_NAME")
    password = os.getenv("MONGO_DB_ATLAS_USER_PASSWORD")
    db_name = os.getenv("MONGO_DB_ATLAS_CLUSTER_NAME")

    uri = f"mongodb+srv://{username}:{password}@cluster0.osiqvwu.mongodb.net/?retryWrites=true&w=majority&appName={db_name}"
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    db_name = "llm"
    collection_name = "little_prince"
    index_name = "vector_index"
    collection = client[db_name][collection_name]

    collection.delete_many({})

    _ = MongoDBAtlasVectorSearch.from_documents(docs, embeddings, collection=collection, index_name=index_name)
    print("Indexed documents into MongoDB Atlas.")


if __name__ == '__main__':
    vectorize(file_path="./data/little_prince.txt")
