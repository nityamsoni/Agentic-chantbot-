import streamlit as st
from langchain_backend import chatbot
from langchain_core.messages import SystemMessage, HumanMessage , BaseMessage


CONFIG={    
    "configurable": {
        "thread_id": "1"
}}




if "message_history" not in st.session_state:
    st.session_state.message_history = []

message_history = st.session_state.message_history


for meassage in message_history:
    with st.chat_message(meassage["role"]):
        st.text(meassage["content"])




user_input =st.chat_input("Type a message...")


if user_input:


    message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    ##  without streaming 

    # response = chatbot.invoke(
    #     {"messages": [HumanMessage(content=user_input)]},
    #     config=CONFIG
    # )
    # assistant_response=response["messages"][-1].content
    # message_history.append({"role": "assistant", "content": assistant_response})
    # with st.chat_message("assistant"):
    #     st.text(assistant_response)

     ##  with streaming 


    with st.chat_message("assistant"):
      assistant_response =   st.write_stream(
            message_chunk.content for message_chunk , metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                         config=CONFIG,
                         stream_mode='messages'
            )
        )

    message_history.append({"role": "assistant", "content": assistant_response})







