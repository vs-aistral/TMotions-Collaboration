import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader

# Defining the prompt template
prompt = ChatPromptTemplate.from_template("Understand the context of the entire article, generate detailed contextual tags from this article: {content}, only give tags, no elaborate descriptions are required")

# Function to initialize the LLM model
def initialize_llm(model_name):
    return ChatOllama(
        model=model_name,
        keep_alive=-1,
        temperature=0,
        max_new_tokens=512
    )

# Define the function to fetch content and generate tags from URL
def generate_tags_from_url(url, llm):
    # Load content from the URL
    loader = WebBaseLoader(url)
    content = loader.load()
    
    # Combining the prompt with the content
    chain = prompt | llm | StrOutputParser()
    
    # Invoking the chain with the fetched content
    return chain.invoke({'content': content})

# Function to generate tags from text input
def generate_tags_from_text(text, llm):
    # Combining the prompt with the content
    chain = prompt | llm | StrOutputParser()
    
    # Invoking the chain with the fetched content
    return chain.invoke({'content': text})

# Integrating backend with a Streamlit app
st.title("Article Tag Generator")

# Sidebar inputs
st.sidebar.header("Input Options")
input_option = st.sidebar.selectbox("Choose input method", ["Enter URL", "Enter Text"])

# LLM Model selection
model_name = st.sidebar.selectbox("Choose LLM Model", ["llama3", "llama2", "phi3:mini","mistral"])

# Initialize the LLM model
llm = initialize_llm(model_name)

if input_option == "Enter URL":
    url_input = st.sidebar.text_input("Enter URL")
    if st.sidebar.button("Generate Tags"):
        if url_input:
            try:
                tags = generate_tags_from_url(url_input, llm)
                st.success("Tags generated successfully!")
                st.write(tags)
            except Exception as e:
                st.error(f"Error generating tags: {e}")
        else:
            st.error("Please enter a valid URL.")
elif input_option == "Enter Text":
    st.header("Input Text")
    text_input = st.text_area("Enter Text", height=300)
    if st.button("Generate Tags"):
        if text_input:
            try:
                tags = generate_tags_from_text(text_input, llm)
                st.success("Tags generated successfully!")
                st.write(tags)
            except Exception as e:
                st.error(f"Error generating tags: {e}")
        else:
            st.error("Please enter valid text.")
