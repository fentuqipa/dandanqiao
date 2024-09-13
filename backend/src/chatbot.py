from typing import Any, Dict, List, Tuple
from langchain_chroma import Chroma
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class CustomHandler(BaseCallbackHandler):
    def __init__(self):
        self.prompt = ""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)
        self.prompt = formatted_prompts

class ChatBot:
    def __init__(self, api_key, is_debug=False):
        self.is_debug = is_debug
        self.model = ChatOpenAI(api_key=api_key)
        self.handler = CustomHandler()
        self.embedding_function = OpenAIEmbeddings(api_key=api_key)
        self.retriever = Chroma(
            embedding_function=self.embedding_function,
            collection_name="documents",
            persist_directory="backend/chroma", 
        ).as_retriever()
    
    def create_chain(self):
        qa_system_prompt = """You are Dandan's assisstant chatbot. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \

        {context}"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompt
        )

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=self.retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(
            # retriever, Replace with History Aware Retriever
            history_aware_retriever,
            chain
        )

        return retrieval_chain

    def process_chat_history(self, chat_history):
        history = []
        for (query, response) in chat_history:
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=response))
        return history

    def generate_response(self, query, chat_history):
        history = self.process_chat_history(chat_history)
        conversational_chain = self.create_chain()
        response = conversational_chain.invoke(
            {
                "input": query,
                "chat_history": history,
            }, 
            config={"callbacks": [self.handler]}
        )["answer"]

        references = self.handler.prompt if self.is_debug else "This is for debugging purposes only."
        return response, references