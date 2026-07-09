from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]

    response = model.invoke(messages)

    return {"messages" : [response]}

# create graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)

# add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Connection
conn = sqlite3.connect(database="06_Chatbot/5_chatbot.db", check_same_thread=False)

# checkpoint
checkpointer = SqliteSaver(conn=conn)

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)

def get_threads():
    all_threads = set()
    for thread in checkpointer.list(None):
        all_threads.add(thread.config['configurable']['thread_id'])
    return list(all_threads)