import asyncio

from fastapi_poe import PoeHandler
from fastapi_poe.types import QueryRequest
from langchain.callbacks import AsyncIteratorCallbackHandler, AsyncCallbackManager
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


class LangChainFastAPIPoeHandler(PoeHandler):
    async def get_response(self, query: QueryRequest):
        # Create a callback manager for this request
        callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        # Create a new LLMChain for this request
        model = ChatOpenAI(callback_manager=callback_manager, streaming=True)
        chain = LLMChain(llm=model, prompt=prompt)

        # Extract the content of the last message
        input = query.query[-1].content

        # Run the chain, we'll await it later
        run = asyncio.create_task(chain.arun(input=input))

        # Yield the tokens as they come in
        async for token in callback_handler.aiter():
            yield self.text_event(token)

        # Await the chain run
        await run
