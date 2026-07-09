import streamlit as st
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from chatbot_backend import chatbot
CONFIG = {"configurable": {"thread_id": "1"}}

load_dotenv()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=CONFIG)['messages'][-1]
    st.session_state["messages"].append({"role": "ai", "content": response.content})
    with st.chat_message("ai"):
        st.write(response.content)