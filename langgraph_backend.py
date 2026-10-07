from langgraph.graph import START , END , StateGraph
from typing import TypedDict , Annotated
from langchain_core.messages import BaseMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
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

checkpointer = InMemorySaver() 

graph = StateGraph(ChatState)
graph.add_node('chat_node' , chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)


