import streamlit as st

# Sidebar
st.sidebar.header("Settings")
enable_smart_chunking = st.sidebar.checkbox("Enable smart chunking")
project_name = st.sidebar.text_input("Project name")
data_uploader = st.sidebar.file_uploader("Data uploader")
query = st.sidebar.text_area("Query", height=200)  # Adjust the height as needed.
top_k = st.sidebar.number_input("top_k", min_value=1, value=5)
if not enable_smart_chunking:
    chunk_size = st.sidebar.number_input("chunk_size", min_value=1, value=256)
    overlap = st.sidebar.number_input("overlap", min_value=0, value=50)
temperature = st.sidebar.slider("temperature", min_value=0.0, max_value=1.0, value=0.5)
generate_button = st.sidebar.button("Generate")

# Main screen
st.title("Smart Chunking")
st.header(f"By: {project_name}")

if generate_button:
    st.write("hello world")
