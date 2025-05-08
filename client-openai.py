import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Callable

import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# nest_asyncio.apply()

# load_dotenv("../.env")

exit_stack = AsyncExitStack()
session: ClientSession = None
stdio = None
write = None

client = AsyncOpenAI(base_url="http://192.168.1.108:80/v1", api_key="token-abc123")


async def connect_to_server(server_script_path: str = "server.py"):
    """Start and connect to the MCP server."""
    global session, stdio, write

    server_params = StdioServerParameters(command="python", args=[server_script_path])
    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()

    tools_result = await session.list_tools()
    print("\n✅ Connected to server with tools:")
    for tool in tools_result.tools:
        print(f"  - {tool.name}: {tool.description}")


async def get_mcp_tools() -> List[Dict[str, Any]]:
    """Fetch tool definitions from the MCP server."""
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


async def get_tool_functions() -> Dict[str, Callable]:
    """Map MCP tool names to async Python callables."""
    tools_result = await session.list_tools()

    async def make_func(tool_name: str):
        async def call_tool(**kwargs):
            result = await session.call_tool(tool_name, kwargs)
            return result.content[0].text if result.content else "No content returned."
        return call_tool

    tool_funcs = {}
    for tool in tools_result.tools:
        tool_funcs[tool.name] = await make_func(tool.name)
    return tool_funcs


async def call_llm(query: str) -> str:
    """Send user query to LLM and handle tool execution if needed."""
    tools = await get_mcp_tools()
    tool_functions = await get_tool_functions()

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can use tools when needed."},
        {"role": "user", "content": query},
    ]

    response = await client.chat.completions.create(
        model="qwen2.5",  
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    print(f"Response: {message.content}")
    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f"\n🛠️ Tool call detected: {tool_name}")
            print(f"📦 Arguments: {json.dumps(arguments, indent=2)}")

            if tool_name in tool_functions:
                print(f"🔧 Executing tool '{tool_name}'...")
                result = await tool_functions[tool_name](**arguments)
                print(f"✅ Tool '{tool_name}' returned: {result}")
            else:
                result = f"❌ Tool '{tool_name}' not found."
                print(result)

            messages.append(message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        print(f"No follow up: {messages}")
        followup = await client.chat.completions.create(
            model="qwen2.5",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return followup.choices[0].message.content
    else:
        return message.content


async def cleanup():
    """Shut down resources cleanly."""
    await exit_stack.aclose()


async def main():
    """Main REPL loop."""
    await connect_to_server()

    while True:
        try:
            query = input("\n💬 Enter your query (or 'exit' to quit): ")
            if query.lower() == "exit":
                print("👋 Exiting...")
                break

            response = await call_llm(query)
            print(f"\n🧠 Response: {response}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
