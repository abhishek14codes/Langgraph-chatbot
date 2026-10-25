from langgraph.graph import START , END , StateGraph
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from rag_tools import rag_search
import requests
import sqlite3
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()


# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     api_key=os.getenv("GEMINI_API_KEY")  
# )
llm = ChatOpenAI(
    model="dots-studio/dots-3-note-preview:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1" 
)

##----tools------
search_tool = DuckDuckGoSearchRun(region="us-en")

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
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=B055OETXOW3UQWHA"
    r = requests.get(url)
    return r.json()

# rag_search is imported from rag_tools.py and added here
tools = [search_tool, get_stock_price, calculator, rag_search]
raw_llm = llm  # keep a reference to the unbound LLM (for summary generation etc.)
llm = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node' , chat_node)
graph.add_node("tools" , tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_threads():
    all_threads = set() 
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return(list(all_threads))
