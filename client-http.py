import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List
import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import streamlit as st

import sys

if sys.platform.startswith("win"):
    # use the ProactorEventLoop on Windows so that create_subprocess_exec works
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


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

    tool_descriptions = "\n".join(
        [f"{t['function']['name']}: {t['function']['description']}" for t in tools]
    )

    system_msg = (
        "You are an assistant. You can call the following tools when needed:\n"
        f"{tool_descriptions}\n"
        "When you need to use a tool, respond naturally using its result.\n"
        "Instead of confirming or clarifying the request with the user, you can directly call the tool and return the result.\n"
        "When using the plot_graph tool, ensure the x and y fields are different and they exist in the column list.\n"
        "Reply only in English.\n"
    )

    if any(t["function"]["name"] == "plot_graph" for t in tools):
        system_msg += (
            "\n[Note on plot_graph] Before you call plot_graph:\n"
            "  • Ensure the file_name is a CSV file.\n"
            "  • Ensure the spec is valid and the columns exist in the data.\n"
            "  • Always invoke get_csv_data_columns on your CSV first.\n"
            "  • Make sure your `x` and `y` fields are different and actually exist in the column list.\n"
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


async def send_query_to_llm(query: str) -> str:
    """Main entry point for the client."""
    await connect_to_server("server.py")
    print(f"\nQuery: {query}")
    response = await process_query(query)
    print(f"\nResponse: {response}")
    await cleanup()
    return response

async def main():
    st.title("Chatbot")
    st.caption("A streamlit chatbot powered by NG YEE TECK")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = await send_query_to_llm(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    print(">>> Client started")
    asyncio.run(main())