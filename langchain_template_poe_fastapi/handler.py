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


class LangChainConversationChainPoeHandler(PoeHandler):
    def __init__(self):
        self.callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([self.callback_handler])
        model = ChatOpenAI(callback_manager=callback_manager, streaming=True)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides "
                "lots of specific details from its context. If the AI does not know the answer to a question, "
                "it truthfully says it does not know."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        memory = ConversationBufferMemory(return_messages=True)  # Can use a different memory here to persist
        self.chain = ConversationChain(memory=memory, prompt=prompt, llm=model)

    async def get_response(self, query: QueryRequest):
        # Get the last message
        chain_input = query.query[-1].content
        print(f"Chain input: {chain_input}")

        # Run the chain, we'll await it later
        run = asyncio.create_task(self.chain.apredict(input=chain_input))

        # Yield the tokens as they come in
        async for token in self.callback_handler.aiter():
            sys.stdout.write(token)
            sys.stdout.flush()
            yield self.text_event(token)

        # Await the chain run
        await run


class LangChainConversationRetrievalChainPoeHandler(PoeHandler):
    """Todo: implement this"""
    pass