from langgraph.graph import StateGraph , START, END
from langchain_core.messages import SystemMessage, HumanMessage , BaseMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict , Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
import operator 
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



from langchain_core.tools import tool
import requests
import os
import sqlite3



load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4.1-Flash",
    task="text-generation",
    provider="novita",
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)




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


tools = [web_search, get_stock_price, calculator]
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
       

conn= sqlite3.connect("chatbot.db" , check_same_thread=False)

checkpointer=SqliteSaver(conn=conn)
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