import streamlit as st
from langgraph_database_backend import chatbot , retrive_threads , raw_llm as llm
from langchain_core.messages import HumanMessage , AIMessage
from rag_tools import ingest_file
import tempfile
import uuid
import os

#*******utility functions***************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return str(thread_id)
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
def load_conversation(thread_id):
    # Ensure thread_id is a string matching your checkpointer format
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    # Return an empty list if 'messages' is not present in values
    return state.values.get('messages', [])
def get_content(content):
    text = ""
    if isinstance(content,str):
        text = content
    elif isinstance(content,list):
        for block in content:
            if isinstance(block,dict) and block.get("type")=="text":
                text += block.get("text","")
    return text 
def gen_summary(thread_id):
        messages= load_conversation(thread_id)
        temp_msg = [] 
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_msg.append({'role':role , 'content':get_content(msg.content)})
        prompt = f'generate a one line summary/title for following conversation {temp_msg}'
        response = llm.invoke(prompt)
        title = get_content(response.content)
        return title

def generate_content(user_input):
    for message , metadata in chatbot.stream(
    {"messages": [HumanMessage(content=user_input)]},
    config=CONFIG,
    stream_mode="messages"
    ) :
        if isinstance(message,AIMessage):
            content = message.content 
            if isinstance(content,str):
                yield content
            elif isinstance(content,list) :
                for block in content:
                    if isinstance(block,dict) and block.get("type") == "text" :
                        yield block.get("text" , "")
    #********session setup**********


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_threads()
add_thread(st.session_state['thread_id'])
if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles']={}

# CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}
CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }
#********Sidebar UI *******************
st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    reset_chat()

# ──── File Upload Section ────────────────────────────────────
st.sidebar.header('📄 Upload Documents')
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files for RAG",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temp location so loaders can read it
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1],
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Ingest into vector store
        with st.sidebar.status(f"Processing {uploaded_file.name}..."):
            result = ingest_file(tmp_path)

        if "error" in result:
            st.sidebar.error(result["error"])
        else:
            st.sidebar.success(
                f"✅ {result['file']}: {result['new_chunks_added']} new chunks added"
            )

        # Clean up temp file
        os.unlink(tmp_path)

st.sidebar.header('My conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    title = st.session_state['chat_titles'].get(str(thread_id),str(thread_id))
    if st.sidebar.button(title,key=f"btn_{thread_id}"): 
        st.session_state['thread_id'] = str(thread_id)
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
    current_thread = str(st.session_state['thread_id'])
    st.session_state['message_history'].append({'role':'user' , 'content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
   
   
    with st.chat_message('assistant'):
        ai_message = st.write_stream(generate_content(user_input))

    st.session_state['message_history'].append({'role':'assistant' , 'content': ai_message})

    if current_thread not in st.session_state['chat_titles']:
        title = gen_summary(current_thread)
        st.session_state['chat_titles'][current_thread] = title 
        st.rerun()
