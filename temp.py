from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(
    model="QwQ-32B",                   
    openai_api_key="not-needed",
    openai_api_base="http://192.168.1.222:1999/v1",
    temperature=0.2
)

help(chat_model.bind_tools)