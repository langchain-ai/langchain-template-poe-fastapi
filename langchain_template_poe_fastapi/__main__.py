from fastapi_poe import run

from .handler import LangChainChatModelPoeHandler, LangChainConversationChainPoeHandler

# run(LangChainChatModelPoeHandler())
run(LangChainConversationChainPoeHandler())
