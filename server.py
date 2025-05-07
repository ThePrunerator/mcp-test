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
def llm_plot_graph(query : str) -> None:
    """Plot a graph using the provided x and y coordinates."""

    CSV_PATH = "test.csv"                     
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.replace('.', '_', regex=False)

    # Set up SQL ──────────────────────────────────────────────────────────────────
    conn = sqlite3.connect("data.db")
    df.to_sql("data", conn, index=False, if_exists="replace")

    # Set up LLM ──────────────────────────────────────────────────────────────────
    llm = ChatOpenAI(
        model="qwen2.5",                   
        openai_api_key="not-needed",
        openai_api_base="http://192.168.1.222/vllm_qwen2.5/v1",
        temperature=0.5,
    )

    # Specify schema for LLM output ──────────────────────────────────────────────────────────────────
    class RequestSpec(BaseModel):
        sql: str  = Field(..., description="Valid SQLite query on table 'data'")
        chart: str = Field(..., description="count | bar | hist | box | scatter | line") # Available chart types
        x: str
        y: str 
        hue: str | None = None
        filter_expr: str | None = Field(
            None,
            description="Optional pandas-query filter applied *after* the SQL.",
        )

    parser = PydanticOutputParser(pydantic_object=RequestSpec) # To convert LLM's JSON output into Python object

    schema_txt = ", ".join(f"{c} ({t})" for c, t in zip(df.columns, df.dtypes))
    format_instr = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    # Prompt fed into LLM ──────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{% raw %}"
                "You are an assistant that turns a user request into:\n"
                "  • a single SQL query for the SQLite table named 'data'\n"
                "  • a JSON spec for plotting with seaborn/matplotlib, following the schema below.\n"
                "  • The x and y labels are mandatory, and they cannot be the same. \n"
                "  • In any case where there are less than 2 fields specified, decide on a suitable value as the corresponding x/y-field. \n"
                f"  • Take note: the only available columns of data are {df.columns}. \n"
                f"  • Take note: the corresponding value types are {df.dtypes}. \n"

                f"This is the schema that the table uses: {schema_txt}\n\n"
                f"{format_instr}"
                "{% endraw %}",
            ),
            ("human", "{{ user_request }}"),
        ],
        template_format="jinja2",
    )

    chain = prompt | llm | parser  # Chain of LLM calls 

    spec: RequestSpec = chain.invoke({"user_request": query})

    print("\n🔎 LLM-generated SQL:\n", spec.sql)
    print("\n🎨 Plot spec:\n", spec)

    data = pd.read_sql_query(spec.sql, conn)
    if spec.filter_expr:
        data = data.query(spec.filter_expr)

    # Plotting of requested graph──────────────────────────────────────────────────────────────────
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