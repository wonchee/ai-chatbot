import os
from typing import Iterator, List, Tuple
import yaml
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from services.chroma_service import ChromaService
from utils import logger


load_dotenv()
chroma_service = ChromaService()
PROMPT_TEMPLATE_PATH = os.path.join("src", "config", "prompt-template.yaml")
DATA_FOLDER_PATH = os.path.join(os.getcwd(), "data")
OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo').lower()

# Load the YAML config file    
with open(PROMPT_TEMPLATE_PATH, 'r') as f:
    config = yaml.safe_load(f)

    
def init_data_directory() -> None:
    """Create data folder for storing uploaded files if it doesn't exist"""
    # Create the directory if it doesn't exist
    os.makedirs(DATA_FOLDER_PATH, exist_ok=True)
    
    
def isDocumentUploaded(file_path: str, file_id: str) -> bool:
    """Check if the document has been uploaded

    Args:
        file_path (str): File path of the document
        file_id (str): File id of the document

    Returns:
        bool: A flag to indicate if the document has been uploaded
    """
    processed_data = st.session_state.processed_data
    
    if file_path in processed_data["file_paths"]:
        return True
    
    if file_id in processed_data["file_ids"]:
        return True
    
    return False
    
    
def get_context(user_query: str) -> Tuple[str, List[str]]:
    """Get the context for LLM prompt

    Args:
        user_query (str): User query
    """
    db = chroma_service.vector_db
    
    # Perform similarity search in vector database to return top 3 most similar chunks
    results = db.similarity_search_with_score(user_query, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return context_text, sources


def get_response(user_query: str, context: str) -> Iterator[str]:
    """Get query response

    Args:
        user_query (str): User query
        context (str): Context for the prompt

    Yields:
        Iterator[str]: Parts of a response as they become available.
    """

    prompt = ChatPromptTemplate.from_template(config['prompt_template'])
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "context": context,
        "question": user_query
    })

# Create data folder if it doesn't exist
init_data_directory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {
        "file_ids": [],
        "file_paths": []
    }
    
st.set_page_config(page_title="AI Chatbot", page_icon=":material/robot_2:")
st.title("AI Chatbot")

file_paths = []
file_ids = []
source_list = ""
 
with st.sidebar:
        uploaded_files = st.file_uploader("Please upload PDF files", accept_multiple_files=True, type=["pdf"])
        
        if uploaded_files:
            with st.spinner('Processing...'):
                for uploaded_file in uploaded_files:
                    file_id = uploaded_file.file_id
                    file_name = uploaded_file.name
                    # Get the full file path of the uploaded file
                    file_path = os.path.join(os.getcwd(), "data", file_name)
                    
                    if file_path in file_paths:
                        continue
                    
                    if isDocumentUploaded(file_path, file_id):
                        continue
                    
                    logger.info("Destination of the uploaded file: " + file_path)

                    # Save the uploaded file to disk
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    file_paths.append(file_path)
                    file_ids.append(file_id)
                
                if len(file_paths) != 0:        
                    all_documents = chroma_service.load_documents(file_paths)
                    chroma_service.add_documents(all_documents)
                    
                    st.session_state.processed_data["file_paths"].extend(file_paths)
                    st.session_state.processed_data["file_ids"].extend(file_ids)
                   

    
# Display the chat history for the same session
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
            if "sources" in message.kwargs and message.kwargs["sources"] != "":
                st.markdown("---")
                st.markdown("*List of sources for context:*")
                st.markdown(message.kwargs["sources"])
                
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
            
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    # Show user query
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Show AI response
    with st.chat_message("AI"):
        # check if the vector database is empty
        has_no_documents = chroma_service.has_no_documents()
        if has_no_documents:
            response = "Please upload at least a file to start the conversation."
            st.write(response)
            
        else:
            context, sources = get_context(user_query)
            response = st.write_stream(get_response(user_query, context))
            
            # Display sources for the response
            if len(sources) > 0:    
                source_list = f"\n- {"\n- ".join(sources)}"
                st.markdown("---")
                st.markdown("*List of sources for context:*")
                st.markdown(source_list)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response, kwargs={
        "sources": source_list
    }))