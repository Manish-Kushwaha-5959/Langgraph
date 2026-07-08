from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

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

# checkpoint
checkpointer = InMemorySaver()

# compile graph
chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":

    config = {"configurable" : {"thread_id" : "1"}}
    # while True:
    #     user_input = input("User: ")

    #     if user_input.strip().lower() in ["exit", "quite", "bye"]:
    #         break

    #     response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    #     print(f"AI: {response['messages'][-1].content}")

    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content="write a 500 words blog on Vollyball")]},
        config= config,
        stream_mode= "messages"
    ):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)


