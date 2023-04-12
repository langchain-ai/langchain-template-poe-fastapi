# langchain-template-poe-fastapi

## Quick Start

Install dependencies
```commandline
poetry install
```

Run the server. By default, the `LangChainChatModelPoeHandler` will be used, but others can be used by setting the `POE_HANDLER` environment variable (see `__main__.py` for more details).
```commandline
make start
```

Make a request
```commandline
curl -X 'POST' \
'http://0.0.0.0:8080/' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{ \
    "version": "1.0", \
    "type": "query", \
    "query": [ \
            { \
                    "message_id": "1", \
                    "role": "system", \
                    "content": "You are a helpful assistant.", \
                    "content_type": "text/markdown", \
                    "timestamp": 1678299819427621, \
                    "feedback": [] \
            }, \
            { \
                    "message_id": "2", \
                    "role": "user", \
                    "content": "What is the capital of Nepal?", \
                    "content_type": "text/markdown", \
                    "timestamp": 1678299819427621, \
                    "feedback": [] \
            } \
    ], \
    "user_id": "u-1234abcd5678efgh", \
    "conversation_id": "c-jklm9012nopq3456", \
    "message_id": "2" \
}' -N
```
