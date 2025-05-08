import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List
import pandas as pd
import re, os
import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

session = None
exit_stack = AsyncExitStack()
stdio = None
write = None

LOCAL_LLM_URL = "http://192.168.1.108:80/v1/chat/completions"
TOKEN = "token-abc123"
LOCAL_LLM_MODEL = "qwen2.5"

async def connect_to_server(server_script_path: str = "server.py"):
    """
    Connect to a MCP server.

    :param server_script_path: Where the client can connect to the server at.
    """
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
    """Recursively handle LLM tool calls until no more are needed."""
    async with aiohttp.ClientSession() as client:
        while True:
            # Step 1: Ask the LLM
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

            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tools needed — return final response
                return message["content"]

            # Step 2: Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                print(f"\nLLM called tool: {tool_name} with arguments: {tool_args}")

                result = await session.call_tool(tool_name, tool_args)
                print(f"Tool result: {result}")

                result_text = result.content[0].text if result.content else "Tool returned no content."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                })

async def process_query(query: str) -> str:
    """Process a query using a local LLM and available MCP tools."""
    global session

    tools = await get_mcp_tools()

    m = re.search(r'\bfile\s*[:=]?\s*([^\s,;]+\.csv)\b', query, re.IGNORECASE)
    file_name = m.group(1) if m else "test.csv"

    file_path = os.path.join("data", file_name)
    try:
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()
    except Exception:
        cols = []

    tools = await get_mcp_tools()

    tool_descriptions = "\n".join(
        [f"{t['function']['name']}: {t['function']['description']}" for t in tools]
    )

    system_msg = (
        "You are an assistant. You can call the following tools when needed:\n"
        f"{tool_descriptions}\n"
        "When you need to use a tool, respond naturally using its result.\n"
        f"When using the plot_graph tool, the file_name must be a CSV file.\n"
        f"When using the plot_graph tool, ensure that the spec is valid. "
        f"When using the plot_graph tool, ensure that the x-field and y-field are parsed, and that they are not the same as each other. "

        f"The following columns are available in `{file_name}`: {', '.join(cols)}\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
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
    
