import streamlit as st

st.set_page_config(
    page_title="Minimal Streamlit App",
    page_icon="📊",
    layout="wide",
)

st.title("Minimal Streamlit App")
st.write("This is a minimal Streamlit app to test if we can run without the inotify error.")

# Simple input and display
user_input = st.text_input("Enter some text")
if user_input:
    st.write(f"You entered: {user_input}")

# Simple button
if st.button("Click me"):
    st.success("Button clicked!")
