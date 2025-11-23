import streamlit as st
from langchain_core.messages import HumanMessage
from rag_model import build_rag_agent_from_file
import tempfile

st.set_page_config(page_title="RAG Assistant", page_icon="📘", layout="wide")

st.title("📘 RAG Question Answering Assistant")
st.write("Upload a TXT or PDF file and ask questions about its content.")

if st.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = []
    st.rerun()

uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])

if uploaded_file:
    with st.spinner("Processing the file..."):
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        rag_agent = build_rag_agent_from_file(temp_path)

    st.success("File processed successfully! You can now ask questions.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.spinner("Thinking..."):
            result = rag_agent.invoke({"messages": [HumanMessage(content=user_input)]})
            answer = result["messages"][-1].content

        st.session_state["messages"].append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)

else:
    st.info("Please upload a TXT or PDF file to get started.")
