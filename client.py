import asyncio
import nest_asyncio
import pandas as pd
import matplotlib.pyplot as plt

# MCP-related libraries
from mcp import ClientSession
from mcp.client.sse import sse_client

# All langchain-related libraries
from langchain_openai import ChatOpenAI
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
        model="QwQ-32B",
        openai_api_key="not-needed",
        openai_api_base="http://192.168.1.222:1999/v1",
        temperature="0.7",
    )

    model_with_tools = chat_model.bind_tools(
        tools = openai_tools,
        tool_choice="auto",
        strict=False
    )

    return model_with_tools

async def send_query_to_llm(csv_data):
    content = f"With reference to the following CSV data columns, please help me plot a relevant graph to better visualise the results.\n \n{csv_data.columns}"
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            model = await setup_llm(session)
            response = model.invoke([HumanMessage(content=content)])
            print(response.content)

def read_csv_data(file_path: str) -> str:
    '''
    Reads CSV file data and returns the output as a DataFrame.
    '''
    data = pd.read_csv(file_path)
    data.columns.str.replace('.', '_', regex=False)  # Replace '.' with '_' in column names
    data.columns.str.replace(' ', '_', regex=False)  # Replace ' ' with '_' in column names

    return data

######################################### Main #########################################
if __name__ == "__main__":
    file_path = ".\\test.csv"
    data = read_csv_data(file_path)
    asyncio.run(send_query_to_llm(data))