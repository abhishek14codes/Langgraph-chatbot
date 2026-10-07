import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


CONFIG = {'configurable':{'thread_id':'thread-1'}}


user_input = st.chat_input('Type Here')


def generate_content(user_input):
    for message , metadata in chatbot.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config=CONFIG,
    stream_mode="messages"
    ) :
        content = message.content 
        if isinstance(content,str):
            yield content
        elif isinstance(content,list) :
            for block in content:
                if isinstance(block,dict) and block.get("type") == "text" :
                    yield block.get("text" , "")






if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

if user_input:

    st.session_state['message_history'].append({'role':'user' , 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
   
   
    with st.chat_message('assistant'):
        ai_message = st.write_stream(generate_content(user_input))

    st.session_state['message_history'].append({'role':'assistant' , 'content': ai_message})
    