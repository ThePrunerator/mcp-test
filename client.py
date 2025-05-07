import asyncio
import nest_asyncio

# MCP-related libraries
from mcp import ClientSession
from mcp.client.sse import sse_client

# All langchain-related libraries
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import HumanMessage

nest_asyncio.apply()  # Needed to run interactive python

######################################### Functions #########################################
async def setup_llm(session: ClientSession) -> ChatOpenAI:
    '''
    Set up a connection to the QwQ-32B model with tools loaded.

    :param session: Session that server is loaded into.
    :return: Model object to establish communication between user and LLM.
    '''
    all_tools = (await session.list_tools()).tools

    openai_tools = []
    for t in all_tools:
        # build a proper OpenAI function schema
        params = {
            "title":       t.name,
            "description": t.description,
            **t.inputSchema,
        }
        openai_tools.append({
            "name":       t.name,
            "description":t.description,
            "parameters":params,
        })

    chat_model = ChatOpenAI(
        model="qwen2.5",
        openai_api_key="not-needed",
        openai_api_base="http://192.168.1.108:80/v1",
        temperature="0.7",
    )

    model_with_tools = chat_model.bind_tools(
        tools = openai_tools,
        tool_choice="auto",
        strict=False
    )

    return model_with_tools

async def main():
    content = input("Enter message: ")

    while content != "END":
        async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                model = await setup_llm(session)
                response = model.invoke([HumanMessage(content=content)])
                print(response.content)
                content = input("Enter message: ")

######################################### Main #########################################
if __name__ == "__main__":
    asyncio.run(main())