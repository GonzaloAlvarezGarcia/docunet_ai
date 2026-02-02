import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_engine import GraphRAGManager
from src.utils import check_neo4j_connection

# Page Config
st.set_page_config(page_title="DocuNet AI", page_icon="🧠", layout="wide")

# Load environment variables
load_dotenv()


def main():
    st.title("🧠 DocuNet AI Assistant")
    st.markdown("### Professional Document Analysis with Graph Knowledge")

    # Sidebar: Configuration and Status
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Connection Check
        if st.button("Check Database Connection"):
            if check_neo4j_connection():
                st.success("✅ Neo4j Connected")
            else:
                st.error("❌ Connection Failed")

        st.divider()

        # File Upload
        uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

        if uploaded_file is not None:
            if st.button("Process & Index Document"):
                with st.spinner("Processing PDF and Building Graph Index..."):
                    try:
                        rag_manager = GraphRAGManager()
                        # 1. Extract text
                        chunks = rag_manager.process_pdf(uploaded_file)
                        # 2. Index into Neo4j
                        rag_manager.index_documents(chunks)
                        st.session_state["rag_ready"] = True
                        st.success(f"✅ Indexed {len(chunks)} chunks successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # Main Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        if st.session_state.get("rag_ready"):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    rag_manager = GraphRAGManager()
                    qa_chain = rag_manager.get_qa_chain()

                    response = qa_chain.invoke({"input": prompt})

                    full_response = response["answer"]

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                except Exception as e:
                    st.error(f"Error generating response: {e}")
        else:
            st.warning("⚠️ Please upload and index a document first.")


if __name__ == "__main__":
    main()
