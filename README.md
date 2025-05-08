<h1 align="center">🧠 LLM Tool-Calling Assistant with MCP Integration</h1>
<p align="center">
  <b>Connect your local LLM to real-world tools, knowledge bases, and APIs via MCP.</b>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/MCP%20Support-Enabled-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/LLM%20Backend-OpenAI%20or%20Local-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Tool%20Calling-Automated-ff69b4?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8+-yellow?style=flat-square" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/74038190/225813708-98b745f2-7d22-48cf-9150-083f1b00d6c9.gif" width="450">
</p>


This project connects a local LLM (e.g. Qwen) to tools such as a calculator or a knowledge base via the [MCP](https://github.com/modelcontextprotocol) protocol. The assistant automatically detects and calls these tools to help answer user queries.

---

## 📦 Features

- 🔧 Tool execution through MCP server  
- 🧠 Local LLM integration via HTTP or OpenAI SDK  
- 📚 Knowledge base support (`data.json`)  
- ⚡ Supports `stdio` and `sse` transports  

---

## 🗂 Project Files

| File              | Description                                                |
|-------------------|------------------------------------------------------------|
| `server.py`       | Registers tools and starts MCP server                      |
| `client-http.py`  | Uses `aiohttp` to communicate with local LLM               |
| `clientopenai.py` | Uses OpenAI-compatible SDK for LLM + tool call logic       |
| `client-stdio.py` | MCP client using stdio                                     |
| `client-see.py`   | MCP client using SSE                                       |
| `data.json`       | Q&A knowledge base                                         |

---

## 📥 Installation

### Requirements

Python 3.8+

Install dependencies:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
aiohttp==3.11.18
nest_asyncio==1.6.0
python-dotenv==1.1.0
openai==1.77.0
mcp==1.6.0
```

---

## 🚀 Getting Started

### 1. Run the MCP server

```bash
python server.py
```

This launches your tool server with functions like `add`, `multiply`, and `get_knowledge_base`.

### 2. Start a client

#### Option A: HTTP client (local LLM via raw API)

```bash
python client-http.py
```

#### Option B: OpenAI SDK client

```bash
python client-openai.py
```

#### Option C: stdio transport

```bash
python client-stdio.py
```

#### Option D: SSE transport

Make sure `server.py` sets:

```python
transport = "sse"
```

Then run:

```bash
python client-sse.py
```

---

## 💬 Example Prompts

### Math Tool Call

```
What is 8 times 3?
```

Response:

```
Eight times three is 24.
```

### Knowledge Base Question

```
What are the healthcare benefits available to employees in Singapore?
```

Response will include the relevant answer from `data.json`.

---

## 📁 Example: `data.json`

```json
[
  {
    "question": "What is Singapore's public holiday schedule?",
    "answer": "Singapore observes several public holidays..."
  },
  {
    "question": "How do I apply for permanent residency in Singapore?",
    "answer": "Submit an online application via the ICA website..."
  }
]
```

---

## 🔧 Configuration

Inside `client-http.py` or `clientopenai.py`, update the following:

```python
LOCAL_LLM_URL = "..."
TOKEN = "your-api-token"
LOCAL_LLM_MODEL = "your-model"
```

Make sure your LLM is serving OpenAI-compatible API endpoints.

---

## 🧹 Cleanup

Clients handle tool calls and responses automatically. You can stop the server or client using `Ctrl+C`.

---

## 🪪 License

MIT License. See [LICENSE](LICENSE) file.
