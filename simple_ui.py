import streamlit as st
import requests

BASE_URL = "http://localhost:8003"  # Backend endpoint

st.set_page_config(
    page_title="Simple Data Analytics Bot",
    page_icon="📊",
    layout="wide",
)

st.title("Simple Data Analytics Bot")
st.write("A simplified version of the Data Analytics Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Simple file uploader
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if st.button("Upload and Process"):
    if uploaded_files:
        files = []
        for f in uploaded_files:
            file_data = f.read()
            if not file_data:
                continue
            files.append(("files", (getattr(f, "name", "file.csv"), file_data, f.type)))

        if files:
            try:
                with st.spinner("Uploading and processing files..."):
                    response = requests.post(f"{BASE_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success("Files uploaded and processed successfully!")
                    else:
                        st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Some files were empty or unreadable.")

# Display chat history
st.subheader("Chat")
for chat in st.session_state.messages:
    if chat["role"] == "user":
        st.write(f"You: {chat['content']}")
    else:
        st.write(f"Bot: {chat['content']}")

# Simple chat input
user_input = st.text_input("Your question")
if st.button("Send") and user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Send to backend
    try:
        with st.spinner("Bot is thinking..."):
            payload = {"question": user_input}
            response = requests.post(f"{BASE_URL}/query", json=payload)
        
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
            st.session_state.messages.append({"role": "bot", "content": answer})
            st.experimental_rerun()
        else:
            st.error(f"Bot failed to respond: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
