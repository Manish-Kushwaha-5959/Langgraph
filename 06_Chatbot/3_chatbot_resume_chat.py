import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot
import uuid

# ************************** utility function *********************************
def generate_threadid():
    thread_id = uuid.uuid4()
    return thread_id

def reset():
    st.session_state["thread_id"] = generate_threadid()
    add_threads(st.session_state["thread_id"])
    st.session_state["messages"] = []

def add_threads(thread_id):
    if thread_id not in st.session_state["threads"]:
        st.session_state["threads"].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state({"configurable": {"thread_id": thread_id}}).values['messages']

# *************************** session *******************************************

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_threadid()

if "threads" not in st.session_state:
    st.session_state["threads"] = []
    add_threads(st.session_state["thread_id"])


# ************************** sidebar *******************************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset()

st.sidebar.header("My Conversations")
for thread in st.session_state["threads"][::-1]:
    if st.sidebar.button(str(thread)):
        messages = load_conversation(thread)
        st.session_state["thread_id"] = thread
        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = 'ai'
            temp_messages.append({'role': role, 'content': msg.content})
        st.session_state["messages"] = temp_messages


# ****************************** UI ***********************************************

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here")
CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("ai"):
        response = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
    st.session_state["messages"].append({"role": "ai", "content": response})
    