import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid


#*******utility functions***************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable':{'thread_id':thread_id}}).values['messages']
def get_content(content):
    if isinstance(content,str):
        text = content
    elif isinstance(content,list):
        for block in content:
            if isinstance(block,dict) and block.get("type")=="text":
                text = block.get("text","")
    return text 

#********session setup**********

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
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
add_thread(st.session_state['thread_id'])

CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}

#********Sidebar UI *******************
st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)): 
        st.session_state['thread_id'] = thread_id
        messages= load_conversation(thread_id)
        temp_msg = [] 
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_msg.append({'role':role , 'content':get_content(msg.content)})
        st.session_state['message_history'] = temp_msg



#************MAIN UI**********************
user_input = st.chat_input('Type Here')

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
