import asyncio
import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatOpenAI(
    model="dots-studio/dots-3-note-preview:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Use sys.executable to point to the current virtual environment's python.exe
client = MultiServerMCPClient(
    {
        "airth": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [r"D:\projects\langgraph-chatbot\math_server.py"],
        }
    }
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await client.get_tools()
    print("Discovered MCP tools:", tools)

    # Bind tools to a new variable name to avoid UnboundLocalError
    model_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile()


async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="find the addition of 221442 and 23 and give answer like a football commentator"
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())