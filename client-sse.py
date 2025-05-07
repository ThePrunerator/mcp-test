import asyncio
import nest_asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

nest_asyncio.apply()  # Needed to run interactive python

async def main():
    # 1. Establish an HTTP SSE connection to your MCP server
    #    sse_client yields a pair of async streams: one for reading SSE events, one for writing commands.
    async with sse_client("http://localhost:8050/sse") as (read_stream, write_stream):

        # 2. Wrap those streams in an MCP ClientSession
        async with ClientSession(read_stream, write_stream) as session:

            # # 3. Perform the MCP handshake / initialization
            # #    This sends the “initialize” JSON to the server so it can register your client
            await session.initialize()
            
            # 4. Ask the server which tools it exposes
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                # Each tool has a .name and .description field
                print(f"  - {tool.name}: {tool.description}")

            # 5. Call the “add” calculator tool, passing the two operands a=2 and b=3
            #    session.call_tool sends a JSON payload like { "tool": "add", "arguments": { "a":2, "b":3 } }
            result = await session.call_tool("add", arguments={"a": 2, "b": 3})

            # 6. The server responds with a JSON-wrapped result. 
            #    result.content is a list of messages; [0].text contains the textual output.
            print(f"2 + 3 = {result.content[0].text}")

"""
Make sure:
1. The server is running before running this script.
2. The server is configured to use SSE transport.
3. The server is listening on port 8050.

To run the server:
uv run server.py
"""

if __name__ == "__main__":
    asyncio.run(main())