import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_database_backend import chatbot, get_threads
import uuid

# ************************** utility function *********************************
def generate_threadid():
    thread_id = uuid.uuid4()
    return str(thread_id)+" new_chat"

def reset():
    st.session_state["thread_id"] = generate_threadid()
    add_threads(st.session_state["thread_id"])
    st.session_state["messages"] = []

def add_threads(thread_id):
    st.session_state["threads"].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state({"configurable": {"thread_id": thread_id}}).values['messages']

# *************************** session *******************************************

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_threadid()

if "threads" not in st.session_state:
    st.session_state["threads"] = get_threads()
    add_threads(st.session_state["thread_id"])


# ************************** sidebar *******************************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    reset()

st.sidebar.header("My Conversations")
for thread in st.session_state["threads"][::-1]:
    title = " ".join(thread.split()[1:])

    if (title == "new_chat"):
        pass

    elif st.sidebar.button(title):
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

NEW_CHAT_CONFIG = {"configurable": {"thread_id": "temp new_chat"}}

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    if len(st.session_state["messages"]) == 1:
        prompt = f"""
Based only on the user's first message, generate a concise chat title.

user first message: {user_input}

Rules:
- Maximum 2–5 words.
- Capture the main topic or intent.
- Use natural title case.
- Do not include quotation marks, emojis, punctuation at the end, or prefixes like "Title:".
- Do not invent details that aren't in the user's message.
- If the message is a greeting (e.g., "Hi", "Hello", "How are you"), return "General Chat".
- Output only the title and nothing else.
"""
        title = chatbot.invoke({"messages": HumanMessage(content=prompt)}, config=NEW_CHAT_CONFIG)["messages"][-1].content
        for i, thread in enumerate(st.session_state["threads"]):
            if thread == st.session_state["thread_id"]:
                thread_id = thread.split()[0] + " " + title
                st.session_state["threads"][i] = thread_id
                st.session_state["thread_id"] = thread_id
                break

    with st.chat_message("user"):
        st.write(user_input)
    
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    with st.chat_message("ai"):
        response = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
        )
    st.session_state["messages"].append({"role": "ai", "content": response})
    st.rerun()
    