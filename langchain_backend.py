from langgraph.graph import StateGraph, START
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Dict, Any, Optional
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ModuleNotFoundError:
    SqliteSaver = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
import os
import tempfile
import sqlite3

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

load_dotenv()
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass





# Tools
@tool
def web_search(query: str) -> str:
    """Search the web for a given query."""
    try:
        return DuckDuckGoSearchRun(region="us-en").run(query)
    except Exception as e:
        return f"Search failed: {e}"

    
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}





@tool
def get_stock_price(symbol: str) -> dict:
    """..."""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=H43T0WRCT38OGZ3A"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}





@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for the current chat thread.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [web_search, get_stock_price, calculator , rag_tool]
llm_with_tools = model.bind_tools(tools)

# -------------------
# 3. State
# -------------------
# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------

class WorkflowState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages ]
def chat_node(state: WorkflowState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


    
def chat_message(state:WorkflowState) -> WorkflowState:
       messages = state['messages']
       response=llm_with_tools.invoke(messages)
       return {'messages': [response]}
       

if SqliteSaver is not None:
    conn = sqlite3.connect("chatbot.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
else:
    # Fallback keeps app booting on platforms where sqlite checkpoint extras are missing.
    conn = None
    checkpointer = MemorySaver()
stategraph=StateGraph(WorkflowState)

stategraph.add_node("chat_node",chat_node)
stategraph.add_node("tools", tool_node)

stategraph.add_edge(START,"chat_node"),
stategraph.add_conditional_edges("chat_node",tools_condition),
stategraph.add_edge('tools', 'chat_node')
chatbot=stategraph.compile(checkpointer=checkpointer)


# graph = StateGraph(ChatState)
# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chat_node")

# graph.add_conditional_edges("chat_node",tools_condition)
# graph.add_edge('tools', 'chat_node')

# chatbot = graph.compile(checkpointer=checkpointer)


def retrive_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)



# response = chatbot.invoke({'messages': [HumanMessage(content='What is the capital of india')]}, config={"configurable": {"thread_id": "1"}})
# print(response)

# thread_id='1'
# while True:
#     user_input = input("User: ")
#     if user_input.lower() == "exit":
#         break
#     print(f"User: {user_input}")
#     initial_state = {
#         'messages': [HumanMessage(content=user_input)]
#     }
#     config = {
#         "configurable": {
#             "thread_id": thread_id
#         }
#     }
#     response = chatbot.invoke(initial_state, config=config)
#     print(f"Chatbot: {response['messages'][-1].content}")
# initial_state = {
#     'messages': [HumanMessage(content='What is the capital of india')]
# }

# chatbot.invoke(initial_state)['messages'][-1].content