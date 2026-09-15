import streamlit as st
from langchain_backend import chatbot , retrive_threads
from langchain_core.messages import SystemMessage, HumanMessage , BaseMessage , AIMessage , ToolMessage
import uuid



## --------------------------------utilities ----------------------------------------------------------------


def generate_thread_id():
    return str(uuid.uuid4())



def reser_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_to_chat_thread(thread_id)

    st.session_state.message_history = []

def add_to_chat_thread(thread_id):
    if thread_id not in st.session_state.chat_thread:
        st.session_state.chat_thread.append(thread_id)




def load_converstaion(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    return state.values.get("messages", [])




##--------------------------------session state initialization----------------------------------------

if "message_history" not in st.session_state:
    st.session_state.message_history = []



message_history = st.session_state.message_history


if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()




if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrive_threads()
    add_to_chat_thread(st.session_state.thread_id)








##---------------------------config and thred_id Passing--------------------------------------------


CONFIG={    
    "configurable": {
        "thread_id": st.session_state["thread_id"]
},
    "metadata": {
        "thread_id": st.session_state["thread_id"]
    },
    "run_name": "chat_turn"
}

st.sidebar.title("Chat with LangGraph")


if st.sidebar.button("New Conversation"):
    reser_chat()

st.sidebar.header("Previous Conversations")


for thread_id in st.session_state["chat_thread"][::-1]:
    if st.sidebar.button(thread_id):
        messages=load_converstaion(thread_id)
        st.session_state["thread_id"] = thread_id

        temp_messages=[]
        for message in messages:
            if isinstance(message,HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})
            
            else:
                temp_messages.append({"role": "assistant", "content": message.content})


        st.session_state.message_history = temp_messages





# for thread_id in st.session_state["chat_thread"][::-1]:

#     messages = load_converstaion(thread_id)

#     title = "New Conversation"

#     for message in messages:
#         if isinstance(message, HumanMessage):
#             title = message.content[:30]
#             break

#     if st.sidebar.button(title, key=thread_id):
#         messages = load_converstaion(thread_id)
#         st.session_state["thread_id"] = thread_id

#         temp_messages = []

#         for message in messages:
#             if isinstance(message, HumanMessage):
#                 temp_messages.append({
#                     "role": "user",
#                     "content": message.content
#                 })
#             else:
#                 temp_messages.append({
#                     "role": "assistant",
#                     "content": message.content
#                 })

#         st.session_state.message_history = temp_messages



#-------------------------------- chatbot UI code----------------------------------------------------# 
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
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )



    

    message_history.append({"role": "assistant", "content": ai_message})







