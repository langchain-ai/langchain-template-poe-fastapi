import asyncio
from typing import List
import sys

from fastapi_poe import PoeHandler
from fastapi_poe.types import ProtocolMessage, QueryRequest
from langchain.callbacks import AsyncIteratorCallbackHandler, AsyncCallbackManager
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage, ChatMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader


def convert_poe_messages(poe_messages: List[ProtocolMessage]) -> List[BaseMessage]:
    messages = []
    for poe_message in poe_messages:
        if poe_message.role == "user":
            messages.append(HumanMessage(content=poe_message.content))
        elif poe_message.role == "assistant":
            messages.append(AIMessage(content=poe_message.content))
        elif poe_message.role == "system":
            messages.append(SystemMessage(content=poe_message.content))
        else:
            messages.append(ChatMessage(content=poe_message.content, role=poe_message.role))
    return messages


class LangChainChatModelPoeHandler(PoeHandler):
    """A simple example of using a ChatModel to handle a conversation."""
    async def get_response(self, query: QueryRequest):
        # Create a callback manager for this request
        callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        # Create a new ChatModel for this request
        model = ChatOpenAI(callback_manager=callback_manager, streaming=True)

        # Convert the poe messages to langchain messages
        messages = convert_poe_messages(query.query)

        # Run the model, we'll await it later
        run = asyncio.create_task(model.agenerate([messages]))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)

        # Await the chain run
        await run


CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and provides "
        "lots of specific details from its context. If the AI does not know the answer to a question, "
        "it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


class LangChainConversationChainPoeHandler(PoeHandler):
    """An example of using a ConversationChain to handle a conversation.
    Note that in a real application, you would want to use a database to store the memory for each conversation.
    This assumes that there is one request per conversation_id at a time.
    """
    def __init__(self):
        self.memories = {}  # Use a different memory per conversation_id

    async def get_response(self, query: QueryRequest):
        # Set up the callback handlers and chains
        callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])
        model = ChatOpenAI(callback_manager=callback_manager, streaming=True)

        # Get the memory for this conversation
        memory = self.memories.get(query.conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.memories[query.conversation_id] = memory
        chain = ConversationChain(memory=memory, prompt=CHAT_PROMPT_TEMPLATE, llm=model)

        # Get the last message
        chain_input = query.query[-1].content
        print(f"Chain input: {chain_input}")

        # Run the chain, we'll await it later
        run = asyncio.create_task(chain.apredict(input=chain_input))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            sys.stdout.write(token)
            sys.stdout.flush()
            yield self.text_event(token)

        # Await the chain run
        await run


class LangChainConversationRetrievalChainPoeHandler(PoeHandler):
    """An example of using a ConversationRetrievalChain to handle a conversation.
    This is useful for asking questions about documents.
    """
    def __init__(self):
        self.chat_history = {}  # Use a different chat history per conversation_id

        # Load the documents for the retrieval chain
        # To make this more efficient, you can load the documents once outside of the handler
        loader = TextLoader("state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(documents, embeddings)

    async def get_response(self, query: QueryRequest):
        # Set up the callback handlers and chains
        callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])
        model = OpenAI(callback_manager=callback_manager, streaming=True)

        # Get the chat history for this conversation
        chat_history = self.chat_history.get(query.conversation_id)
        if chat_history is None:
            chat_history = []
            self.chat_history[query.conversation_id] = chat_history

        # Get the last message
        chain_input = query.query[-1].content

        # Run the chain, we'll await it later
        chain = ConversationalRetrievalChain(
            vectorstore=self.vectorstore,
            prompt=CHAT_PROMPT_TEMPLATE,
            llm=model,
            chat_history=chat_history
        )
        run = asyncio.create_task(chain.acall({"question": chain_input, "chat_history": chat_history}))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            sys.stdout.write(token)
            sys.stdout.flush()
            yield self.text_event(token)

        # Await the chain run
        result = await run

        # Add the result to the chat history
        chat_history.append((chain_input, result["answer"]))
