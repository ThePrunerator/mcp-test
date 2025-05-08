from mcp.server.fastmcp import FastMCP
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
from pydantic import BaseModel
from typing import Literal, Optional
import json, asyncio

import sys

if sys.platform.startswith("win"):
    # use the ProactorEventLoop on Windows so that create_subprocess_exec works
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Create a MCP server
mcp = FastMCP(
    name="graph-plotter",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8050,  # only used for SSE transport (set this to any port)
)

@mcp.tool()
def get_csv_data_columns(file_name : str) -> str:
    """Get the data from a CSV file."""
    file_path = os.path.join("data", file_name)
    try:
        data = pd.read_csv(file_path)
        columns = data.columns.tolist()
        return json.dumps(columns)
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

class PlotSpec(BaseModel):
    chart: Literal["count", "bar", "hist", "box", "scatter", "line"]
    x:     str
    y:     Optional[str] = None
    hue:   Optional[str] = None

@mcp.tool()
def find_all_data_files() -> list[str]:
    '''
    Gets all CSV files in the current directory (and subfolders) and returns their full paths.
    '''
    dir = os.getcwd()
    to_return = []

    for folder, subfolder, files in os.walk(dir):
        for file in files:
            if ".csv" in file:
                to_return.append(os.path.join(folder, file))

    return to_return

@mcp.tool()
def plot_graph(file_name : str, spec : PlotSpec, query : str) -> None:
    """
    Plot a graph using the provided queries.
    Output "Image has been created." in your response string
    when a new output.png image has been created.
    """

    file_path = os.path.join("data", file_name)
    data = pd.read_csv(file_path)

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
    plt.savefig("output.png")
    plt.close()

@mcp.tool()
def look_for_csv(file_name: str) -> str:
    '''
    Searches for & returns the full file path for the file name
    input as parameter.

    :param file_name: Name of file to search for.
    :return" Full file path to image.
    '''
    curr_dir = ".\\data"
    for folder, subfolders, files in os.walk(curr_dir):
        for file in files:
            if file == file_name:
                return os.path.join(folder, file)
            
@mcp.tool()
def get_cwd() -> str:
    '''
    Gets the current working directory.
    '''
    return os.getcwd()

# Run the server
if __name__ == "__main__":
    transport = "stdio"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")