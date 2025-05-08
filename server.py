from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os, json
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import pandas as pd
import sqlite3

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv("../.env")

# Create an MCP server
mcp = FastMCP(
    name="Calculator",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

@mcp.tool()
def get_knowledge_base() -> str:
    """Retrieve the entire knowledge base as a formatted string.

    Returns:
        A formatted string containing all Q&A pairs from the knowledge base.
    """
    try:
        kb_path = os.path.join(os.path.dirname(__file__), "data.json")
        with open(kb_path, "r") as f:
            kb_data = json.load(f)

        # Format the knowledge base as a string
        kb_text = "Here is the retrieved knowledge base:\n\n"

        if isinstance(kb_data, list):
            for i, item in enumerate(kb_data, 1):
                if isinstance(item, dict):
                    question = item.get("question", "Unknown question")
                    answer = item.get("answer", "Unknown answer")
                else:
                    question = f"Item {i}"
                    answer = str(item)

                kb_text += f"Q{i}: {question}\n"
                kb_text += f"A{i}: {answer}\n\n"
        else:
            kb_text += f"Knowledge base content: {json.dumps(kb_data, indent=2)}\n\n"

        return kb_text
    except FileNotFoundError:
        return "Error: Knowledge base file not found"
    except json.JSONDecodeError:
        return "Error: Invalid JSON in knowledge base file"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def plot_graph(spec : str, query : str) -> None:
    """Plot a graph using the provided x and y coordinates."""

    plt.figure(figsize=(10, 6))
    if spec.chart == "count":               
        sns.countplot(data=data, x=spec.x, hue=spec.hue,
                    order=data[spec.x].value_counts().index)
    elif spec.chart == "bar":
        sns.barplot(data=data, x=spec.x, y=spec.y, hue=spec.hue)
    elif spec.chart == "hist":
        sns.histplot(data=data, x=spec.x, hue=spec.hue, kde=True)
    elif spec.chart == "box":
        sns.boxplot(data=data, x=spec.x, y=spec.y, hue=spec.hue)
    elif spec.chart == "scatter":
        sns.scatterplot(data=data, x=spec.x, y=spec.y, hue=spec.hue)
    elif spec.chart == "line":
        sns.lineplot(data=data, x=spec.x, y=spec.y, hue=spec.hue)    
    else:
        raise ValueError(f"Unknown chart type: {spec.chart}")

    plt.title(textwrap.fill(query, 60))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run the server
if __name__ == "__main__":
    transport = "sse"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")