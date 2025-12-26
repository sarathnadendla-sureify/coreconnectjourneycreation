import streamlit as st
import requests
import json
import pandas as pd
import time
import html
import io
import zipfile

BASE_URL = "http://127.0.0.1:8000"  # Backend endpoint

st.set_page_config(
    page_title="🛣️ Sureify Journey Generator",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛣️ Sureify Journey Generator")

# Add a description
st.markdown("""
Welcome to the Sureify Journey Generator!

This tool helps you manage and create insurance journeys using Retrieval-Augmented Generation (RAG). Upload multiple journey files (TXT, or TS) containing user flows, events, or structured journey data. Then, ask questions or prompts to generate new journeys based on your existing journey data.

**How it works:**
- Upload your journey data files (TXT, or TS) related to insurance or customer journeys.
- The bot will analyze and index your journeys using RAG.
- Ask questions or prompts to generate new journeys, or to get insights from your existing journeys.
- The system will use your uploaded journeys as context to generate new, relevant journeys tailored to your needs.

**Example prompts:**
- "Create a new onboarding journey for a user based on existing journeys."
- "What are the most common steps in our claim journeys?"
- "Generate a journey for a new insurance product launch."
""")

# Initialize session state
for key in ["index_status", "last_upload_time", "messages", "selected_example_query"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "messages" else []

# Sidebar
with st.sidebar:
    st.header("📄 Upload Data")

    st.subheader("Upload TXT, or TS Files")
    st.markdown("Upload your **TXT, or TS journey data files** to analyze with the bot.")
    uploaded_files = st.file_uploader("Choose files", type=["txt", "ts","tsx"], accept_multiple_files=True)

    if uploaded_files:
        st.subheader("File Preview")
        for f in uploaded_files:
            with st.expander(f"Preview: {f.name}"):
                try:
                    f.seek(0)
                    if f.name.lower().endswith('.txt') or f.name.lower().endswith('.ts') or f.name.lower().endswith('.tsx'):
                        text = f.read().decode('utf-8', errors='replace')
                        st.text_area("Text file preview", text[:2000], height=200)
                    f.seek(0)
                except Exception as e:
                    st.error(f"Could not preview file: {str(e)}")

    if st.button("Upload and Process Data"):
        if uploaded_files:
            files = []
            for f in uploaded_files:
                file_data = f.read()
                if file_data:
                    files.append(("files", (f.name, file_data, f.type)))

            if files:
                try:
                    with st.spinner("Uploading and processing files..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)

                        response = requests.post(f"{BASE_URL}/upload", files=files)
                        if response.status_code == 200:
                            st.success("✅ Files uploaded and processed successfully!")
                            st.session_state.index_status = "ready"
                            st.session_state.last_upload_time = time.time()

                            if not any(msg["role"] == "system" for msg in st.session_state.messages):
                                st.session_state.messages.append({
                                    "role": "system",
                                    "content": "I've processed your data files. You can now ask questions about the data."
                                })
                        else:
                            st.error("❌ Upload failed: " + response.text)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.subheader("Example Queries")
    example_queries = [
        "Create a new onboarding journey for a user based on existing journeys.",
        "What are the most common steps in our claim journeys?",
        "Generate a journey for a new insurance product launch.",
        "What are the most common event types?",
        "Generate Python code to process journey data and extract unique user IDs.",
    ]

    for query in example_queries:
        if st.button(query, key=f"example_{query}"):
            st.session_state.selected_example_query = query
            st.rerun()

# If an example query was selected
if st.session_state.selected_example_query:
    query = st.session_state.selected_example_query
    st.session_state.selected_example_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    try:
        with st.spinner("Bot is thinking..."):
            payload = {"question": query}
            response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
            st.session_state.messages.append({"role": "bot", "content": answer})
        else:
            st.error("❌ Bot failed to respond: " + response.text)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with your Data")

    if st.session_state.index_status == "ready":
        st.success("✅ Data is loaded and ready for queries")
    elif st.session_state.last_upload_time:
        st.info("⏳ Data is being processed...")

    # --- ZIP DOWNLOAD FEATURE ---
    def extract_code_files_from_messages(messages):
        files = {}
        import re
        for chat in messages:
            if chat["role"] == "bot":
                content = chat["content"]
                if isinstance(content, list):
                    content = '\n'.join(str(item) for item in content)
                if '// Filename:' in content:
                    file_sections = content.split('// Filename:')
                    for section in file_sections:
                        section = section.strip()
                        if not section:
                            continue
                        lines = section.split('\n', 1)
                        filename = lines[0].strip() if lines else ''
                        file_content = lines[1] if len(lines) > 1 else ''
                        code_block_pattern = re.compile(r'```(typescript|ts|js|json|md|jsx)?\n?(.*?)```', re.DOTALL)
                        code_block_match = code_block_pattern.search(file_content)
                        if code_block_match:
                            code = code_block_match.group(2)
                            files[filename] = code.strip()
                        elif file_content.strip():
                            files[filename] = file_content.strip()
        return files

    code_files = extract_code_files_from_messages(st.session_state.messages)
    if code_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for fname, fcontent in code_files.items():
                zip_file.writestr(fname, fcontent)
        zip_buffer.seek(0)
        st.download_button(
            label="⬇️ Download All Code as Zip",
            data=zip_buffer,
            file_name="generated_code.zip",
            mime="application/zip",
            use_container_width=True
        )
    # --- END ZIP DOWNLOAD FEATURE ---

    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.messages:
            if chat["role"] == "system":
                st.info(chat["content"])
            elif chat["role"] == "user":
                safe_content = html.escape(str(chat['content']))
                st.markdown(f"<div style='background:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px;'><strong>🧑 You:</strong> {safe_content}</div>", unsafe_allow_html=True)
            else:
                content = chat["content"]
                if isinstance(content, list):
                    content = '\n'.join(str(item) for item in content)
                # Support multiple file outputs by splitting on '// Filename:'
                if '// Filename:' in content:
                    file_sections = content.split('// Filename:')
                    for section in file_sections:
                        section = section.strip()
                        if not section:
                            continue
                        lines = section.split('\n', 1)
                        filename = lines[0].strip() if lines else ''
                        file_content = lines[1] if len(lines) > 1 else ''
                        import re
                        if re.match(r'^[\w\-/]+\.(ts|js|json|md|tsx|jsx)$', filename):
                            st.markdown(f"<div style='font-weight:bold;color:#007acc;margin-top:1em;'>// Filename: {filename}</div>", unsafe_allow_html=True)
                            # Extract the first code block after the filename, regardless of any markdown/explanation
                            code_block_pattern = re.compile(r'```(typescript|ts|js|json|md|jsx)?\n?(.*?)```', re.DOTALL)
                            code_block_match = code_block_pattern.search(file_content)
                            if code_block_match:
                                # Show any explanation/markdown before the code block
                                before = file_content[:code_block_match.start()]
                                if before.strip():
                                    st.markdown(f"<div style='background:#f9f9f9;padding:8px;border-radius:6px;font-style:italic;margin-bottom:6px;'>{before.strip()}</div>", unsafe_allow_html=True)
                                code_lang = code_block_match.group(1) or ''
                                code = code_block_match.group(2)
                                st.markdown("<span style='color:#007acc;font-size:0.9em;'>Click to copy code ⬇️</span>", unsafe_allow_html=True)
                                # Infer language from filename if missing
                                if not code_lang:
                                    if filename.endswith('.ts') or filename.endswith('.tsx'):
                                        code_lang = 'typescript'
                                    elif filename.endswith('.js') or filename.endswith('.jsx'):
                                        code_lang = 'javascript'
                                    elif filename.endswith('.json'):
                                        code_lang = 'json'
                                    elif filename.endswith('.md'):
                                        code_lang = 'markdown'
                                st.code(code.strip(), language=code_lang if code_lang else None)
                                # Download button for the code block
                                st.download_button(
                                    label=f"⬇️ Download {filename}",
                                    data=code.strip(),
                                    file_name=filename,
                                    mime='text/plain',
                                    use_container_width=True
                                )
                                # Show any explanation/markdown after the code block
                                after = file_content[code_block_match.end():]
                                if after.strip():
                                    st.markdown(f"<div style='background:#f9f9f9;padding:8px;border-radius:6px;font-style:italic;margin-bottom:6px;'>{after.strip()}</div>", unsafe_allow_html=True)
                            else:
                                # If no code block, treat the whole file as code for copyability and download
                                if file_content.strip():
                                    st.markdown("<span style='color:#007acc;font-size:0.9em;'>Click to copy code ⬇️</span>", unsafe_allow_html=True)
                                    # Infer language from filename
                                    code_lang = None
                                    if filename.endswith('.ts') or filename.endswith('.tsx'):
                                        code_lang = 'typescript'
                                    elif filename.endswith('.js') or filename.endswith('.jsx'):
                                        code_lang = 'javascript'
                                    elif filename.endswith('.json'):
                                        code_lang = 'json'
                                    elif filename.endswith('.md'):
                                        code_lang = 'markdown'
                                    st.code(file_content.strip(), language=code_lang)
                                    st.download_button(
                                        label=f"⬇️ Download {filename}",
                                        data=file_content.strip(),
                                        file_name=filename,
                                        mime='text/plain',
                                        use_container_width=True
                                    )
                        else:
                            # If not a valid filename, treat the whole section as markdown (for explanations, etc.)
                            if section.strip():
                                st.markdown(f"<div style='background:#f9f9f9;padding:8px;border-radius:6px;font-style:italic;margin-bottom:6px;'>{section.strip()}</div>", unsafe_allow_html=True)
                # Fallback for single code block or plain markdown
                else:
                    import re
                    code_block_pattern = re.compile(r'```(typescript|ts|js|json)?(.*?)```', re.DOTALL)
                    code_blocks = list(code_block_pattern.finditer(content))
                    if code_blocks:
                        last_end = 0
                        for match in code_blocks:
                            before = content[last_end:match.start()]
                            if before.strip():
                                st.markdown(f"<div style='background:#f9f9f9;padding:8px;border-radius:6px;font-style:italic;margin-bottom:6px;'>{before.strip()}</div>", unsafe_allow_html=True)
                            code_lang = match.group(1) or ''
                            code = match.group(2)
                            st.markdown("<span style='color:#007acc;font-size:0.9em;'>Click to copy code ⬇️</span>", unsafe_allow_html=True)
                            st.code(code.strip(), language=code_lang if code_lang else None)
                            last_end = match.end()
                        after = content[last_end:]
                        if after.strip():
                            st.markdown(f"<div style='background:#e6f3ff;padding:10px;border-radius:10px;margin-bottom:10px;'><strong>🤖 Bot:</strong> {html.escape(after)}</div>", unsafe_allow_html=True)
                    else:
                        # If no code block, check if content looks like code (e.g., starts with import, export, function, class, or has curly braces)
                        code_like_pattern = re.compile(r'^(\s)*(import |export |function |class |const |let |var |interface |type |\{|\})', re.MULTILINE)
                        if code_like_pattern.search(content):
                            st.markdown("<span style='color:#007acc;font-size:0.9em;'>Click to copy code ⬇️</span>", unsafe_allow_html=True)
                            st.code(content.strip(), language=None)
                        else:
                            st.markdown(f"<div style='background:#f9f9f9;padding:8px;border-radius:6px;font-style:italic;margin-bottom:6px;'>{content.strip()}</div>", unsafe_allow_html=True)

    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question about the data", placeholder="e.g., List all userids with eventtype 'navigate'")
        col_a, col_b, _ = st.columns([1, 1, 5])
        with col_a:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        with col_b:
            clear_button = st.form_submit_button("Clear Chat", use_container_width=True)

    if clear_button:
        st.session_state.messages = []
        st.rerun()

    if submit_button and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.rerun()

if (
    st.session_state.messages and
    st.session_state.messages[-1]["role"] == "user" and
    not any(msg.get("processing") for msg in st.session_state.messages)
):
    last_msg = st.session_state.messages[-1]
    last_msg["processing"] = True
    try:
        with st.spinner("Bot is thinking..."):
            payload = {"question": last_msg["content"]}
            response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer returned.")
            if isinstance(answer, str):
                if answer.strip().startswith('{') and answer.strip().endswith('}'):
                    try:
                        json_data = json.loads(answer)
                        answer = f"```json\n{json.dumps(json_data, indent=2)}\n```"
                    except:
                        pass
            st.session_state.messages.append({"role": "bot", "content": answer})
        else:
            st.error("❌ Bot failed to respond: " + response.text)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    del last_msg["processing"]
    st.rerun()

with col2:
    st.header("📋 Data Guide")
    if st.session_state.index_status == "ready":
        st.subheader("Your Data")
        st.markdown("""
        - **Status**: Ready for queries
        - **Last updated**: Recently

        You can ask questions about:
        - User IDs and their associated data
        - Event types and their frequency
        - Specific fields and their values
        - Relationships between different data points
        """)

        st.subheader("Query Tips")
        st.markdown("""
        **Effective queries:**
        - Be specific about what you're looking for
        - Mention field names when possible
        - For exact matches, use quotes (e.g., "navigate")
        - Ask one question at a time

        **Sample queries:**
        - "What are all the event types in the data?"
        - "Show me data for user ID 2432a45a-d53f-4da..."
        """)
