import asyncio
from typing import List

from fastapi_poe import PoeHandler
from fastapi_poe.types import ProtocolMessage, QueryRequest
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.schema import (AIMessage, BaseMessage, ChatMessage,
                              HumanMessage, SystemMessage)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def convert_poe_messages(poe_messages: List[ProtocolMessage]) -> List[BaseMessage]:
    """Convert a list of ProtocolMessage to a list of BaseMessage."""
    messages = []
    for poe_message in poe_messages:
        if poe_message.role == "user":
            messages.append(HumanMessage(content=poe_message.content))
        elif poe_message.role == "assistant":
            messages.append(AIMessage(content=poe_message.content))
        elif poe_message.role == "system":
            messages.append(SystemMessage(content=poe_message.content))
        else:
            messages.append(
                ChatMessage(content=poe_message.content, role=poe_message.role)
            )
    return messages


class LangChainChatModelPoeHandler(PoeHandler):
    """A simple example of using a ChatModel to handle a conversation."""

    async def get_response(self, query: QueryRequest):
        # Create a callback manager for this request
        callback_handler = AsyncIteratorCallbackHandler()

        # Create a new ChatModel for this request
        model = ChatOpenAI(callbacks=[callback_handler], streaming=True)

        # Convert the poe messages to langchain messages
        messages = convert_poe_messages(query.query)

        # Run the model as a separate task, we'll await it later
        run = asyncio.create_task(model.agenerate([messages]))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)

        # Await the chain run
        await run


CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a friendly conversation between a human and an AI. The AI is talkative and provides "
            "lots of specific details from its context. If the AI does not know the answer to a question, "
            "it truthfully says it does not know."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


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
        model = ChatOpenAI(callbacks=[callback_handler], streaming=True)

        # Get the memory for this conversation
        memory = self.memories.get(query.conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.memories[query.conversation_id] = memory
        chain = ConversationChain(memory=memory, prompt=CHAT_PROMPT_TEMPLATE, llm=model)

        # Get the last message
        chain_input = query.query[-1].content
        print(f"Chain input: {chain_input}")

        # Run the chain as a separate task, we'll await it later
        run = asyncio.create_task(chain.arun(input=chain_input))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)

        # Await the chain run
        await run


class LangChainConversationRetrievalChainPoeHandler(PoeHandler):
    """An example of using a ConversationRetrievalChain to handle a conversation.
    This is useful for asking questions about documents.
    This assumes that there is one request per conversation_id at a time.

    You will need to install the following packages:
    pip install chromadb
    pip install tiktoken
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
        question_gen_model = OpenAI(temperature=0)
        stream_model = OpenAI(callback_manager=callback_manager, streaming=True)

        # Get the chat history for this conversation
        chat_history = self.chat_history.get(query.conversation_id)
        if chat_history is None:
            chat_history = []
            self.chat_history[query.conversation_id] = chat_history

        # Get the last message
        chain_input = query.query[-1].content

        # Set up the chain
        question_generator = LLMChain(
            llm=question_gen_model, prompt=CONDENSE_QUESTION_PROMPT
        )
        doc_chain = load_qa_chain(
            stream_model,
            chain_type="stuff",
            prompt=QA_PROMPT,
            callback_manager=callback_manager,
        )

        chain = ConversationalRetrievalChain(
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
            retriever=self.vectorstore.as_retriever(),
        )

        # Run the chain as a separate task, we'll await it later
        run = asyncio.create_task(
            chain.acall({"question": chain_input, "chat_history": chat_history})
        )

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)

        # Await the chain run
        result = await run

        # Add the result to the chat history
        chat_history.append((chain_input, result["answer"]))
