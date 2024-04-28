#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Mustafa Durmuş [mustafa.durmus@albert.health]

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Initialize MongoDB python client and get vector store.
username = os.getenv("MONGO_DB_ATLAS_USER_NAME")
password = os.getenv("MONGO_DB_ATLAS_USER_PASSWORD")
cluster_name = os.getenv("MONGO_DB_ATLAS_CLUSTER_NAME")
db_name = "llm"
collection_name = "little_prince"
uri = f"mongodb+srv://{username}:{password}@cluster0.osiqvwu.mongodb.net/?retryWrites=true&w=majority&appName={cluster_name}"
MONGO_VECTOR_STORE = MongoDBAtlasVectorSearch(collection=MongoClient(uri)[db_name][collection_name],
                                              embedding=OpenAIEmbeddings(),
                                              index_name="vector_index")


class LLM:
    def __init__(self, system_prompt: str, model: str,
                 max_tokens: int, top_p: float, frequency_penalty: float,
                 presence_penalty: float,
                 temperature: float):
        self.system_prompt = system_prompt

        # create llm model
        self.llm_model = ChatOpenAI(temperature=temperature, model=model, max_tokens=max_tokens,
                                    model_kwargs={'presence_penalty': presence_penalty,
                                                  'frequency_penalty': frequency_penalty,
                                                  'top_p': top_p})
        self.prompt_template = self.create_prompt_template(system_prompt=system_prompt)

    @staticmethod
    def create_prompt_template(system_prompt: str):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}"),
            ("user", """Now let's continue with the user's query:
                        Question: {input}
                        ===
                        Be short and on the point.
                        <context>
                            {context}
                        </context>
                        ===
                        Answer:""")])
        return prompt_template

    def ask(self, user_prompt: str) -> dict:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | self.prompt_template
                | self.llm_model
                | StrOutputParser()
        )

        # create retriever and get top 2 documents.
        retriever = MONGO_VECTOR_STORE.as_retriever(search_kwargs={'k': 2}, similarity_type="similarity")
        # create a chain and assign source to the chain.
        rag_chain_with_source = RunnableParallel({"context": retriever, "input": RunnablePassthrough()}).assign(
            answer=rag_chain_from_docs)
        return rag_chain_with_source.invoke(user_prompt)


if __name__ == '__main__':
    question = "who have been seen the B-612 asteroid first?"

    my_llm = LLM(system_prompt="you are a helpful assistant.",
                 model="gpt-4-1106-preview", temperature=0.1, max_tokens=256,
                 frequency_penalty=0, presence_penalty=0.6, top_p=1)

    response = my_llm.ask(question)
    for i in response['context']:
        print(i.page_content)
        print("-" * 100)

    print(response['answer'])
