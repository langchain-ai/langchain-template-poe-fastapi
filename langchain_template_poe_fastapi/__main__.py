import os

from fastapi_poe import run

from .handler import (LangChainChatModelPoeHandler,
                      LangChainConversationChainPoeHandler,
                      LangChainConversationRetrievalChainPoeHandler)

handler_str = os.getenv("POE_HANDLER", "default_chat")
if handler_str == "default_chat":
    handler = LangChainChatModelPoeHandler()
elif handler_str == "conversation":
    handler = LangChainConversationChainPoeHandler()
elif handler_str == "conversation_retrieval":
    handler = LangChainConversationRetrievalChainPoeHandler()
else:
    raise ValueError(f"Unknown handler {handler_str}")

run(handler)
