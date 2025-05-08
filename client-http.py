import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List
import pandas as pd
import os

import aiohttp
import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# nest_asyncio.apply()
# load_dotenv("../.env")

session = None
exit_stack = AsyncExitStack()
stdio = None
write = None

LOCAL_LLM_URL = "http://192.168.1.108:80/v1/chat/completions"  # URL of your local LLM API endpoint
TOKEN = "token-abc123"  # Replace with your actual API token
LOCAL_LLM_MODEL = "qwen2.5"  # Specify the name of the local model to use
file_path = os.path.join("data", "test.csv")  # Path to your CSV file
df = pd.read_csv(file_path)  # Load your CSV data here


async def connect_to_server(server_script_path: str = "server.py"):
    """Connect to an MCP server."""
    global session, stdio, write, exit_stack

    server_params = StdioServerParameters(
        command="python",
        args=[server_script_path],
    )
    print("Connecting to server...")
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()

    tools_result = await session.list_tools()
    print("\nConnected to server with tools:")
    for tool in tools_result.tools:
        print(f"  - {tool.name}: {tool.description}")


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get available tools from the MCP server in OpenAI format."""
    global session

    tools_result = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools_result.tools
    ]


async def call_local_llm(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> str:
    """Send a chat request to the local LLM, handle tool calls, and return final response."""
    async with aiohttp.ClientSession() as client:
        # First call to LLM
        response = await client.post(
            LOCAL_LLM_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
            json={
                "model": LOCAL_LLM_MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            },
        )
        data = await response.json()

        if "choices" not in data or not data["choices"]:
            raise ValueError(f"Invalid LLM response: {data}")

        message = data["choices"][0]["message"]
        messages.append(message)

        # Check for tool call
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            print(f"\nLLM called tool: {tool_name} with arguments: {tool_args}")

            # Run the tool using MCP
            tool_result = await session.call_tool(tool_name, tool_args)
            print(f"Tool result: {tool_result}")

            # Extract text from tool result
            tool_output = tool_result.content[0].text if tool_result.content else "Tool returned no content."

            # Send tool result back to model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_output,
            })

            # Second call to LLM with tool result
            followup = await client.post(
                LOCAL_LLM_URL,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"},
                json={
                    "model": LOCAL_LLM_MODEL,
                    "messages": messages,
                    "stream": False,
                },
            )
            final_data = await followup.json()

            if "choices" not in final_data or not final_data["choices"]:
                raise ValueError(f"Invalid final LLM response: {final_data}")

            return final_data["choices"][0]["message"]["content"]
        else:
            return message["content"]


async def process_query(query: str) -> str:
    """Process a query using a local LLM and available MCP tools."""
    global session

    tools = await get_mcp_tools()

    tool_descriptions = "\n".join(
        [f"{t['function']['name']}: {t['function']['description']}" for t in tools]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant. You can call the following tools when needed:\n"
                f"{tool_descriptions}\n"
                "When you need to use a tool, respond naturally using its result.\n"

                f"When using the plot_graph tool, ensure that the spec is valid. The following columns are available: {df.columns} \n"
            ),
        },
        {"role": "user", "content": query},
    ]

    response_text = await call_local_llm(messages, tools)
    return response_text


async def cleanup():
    """Clean up resources."""
    global exit_stack
    await exit_stack.aclose()


async def main():
    """Main entry point for the client."""
    await connect_to_server("server.py")

    while True:
        try:
            query = input("\nEnter your query (or 'exit' to quit): ")

            if query.lower() == 'exit':
                print("Exiting...")
                break

            print(f"\nQuery: {query}")

            response = await process_query(query)
            print(f"\nResponse: {response}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    await cleanup()


if __name__ == "__main__":
    print(">>> Client started")
    asyncio.run(main())
    
