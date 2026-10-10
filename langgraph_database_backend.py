from langgraph.graph import START , END , StateGraph
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
import os
from dotenv import load_dotenv


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")  
)



class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages":[response]}


conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node' , chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_threads():
    all_threads = set() 
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return(list(all_threads))
