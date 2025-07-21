import os
import streamlit as st
from typing import Optional
import pytesseract
from PIL import Image
import PyPDF2
from llama_cpp import Llama
from googletrans import Translator
from llama_pdf import DocumentSummarizer

def main():
    # Streamlit App Configuration
    st.set_page_config(
        page_title="Document Summarizer",
        page_icon="📄",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title and Description
    st.title("📄 Document Summarization Tool")
    st.markdown("### Summarize PDFs and Images using AI")

    # Model Path Input
    # st.sidebar.header("🤖 Model Configuration")
    # model_path = st.sidebar.text_input(
    #     "Llama Model Path", 
    #     value="path/to/your/model.gguf",
    #     help="Provide the full path to your Llama model file"
    # )
    model_path = "D:/PROJECT EXPO 1/Harshgup16/llama-3-8b-Instruct-bnb-4bit-laptop-recommendation/unsloth.Q4_K_M.gguf"

    # File Upload
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file", 
        type=['pdf', 'png', 'jpg', 'jpeg', 'webp']
    )

    # Additional Parameters
    col1, col2 = st.columns(2)
    with col1:
        context_length = st.number_input(
            "Context Length", 
            min_value=1024, 
            max_value=8192, 
            value=4096,
            help="Maximum number of tokens the model can process"
        )
    
    with col2:
        max_summary_tokens = st.number_input(
            "Max Summary Tokens", 
            min_value=100, 
            max_value=1024, 
            value=500,
            help="Maximum number of tokens in the summary"
        )

    # Summarization Process
    if st.button("Generate Summary"):
        # Input Validation
        if not uploaded_file:
            st.error("Please upload a document first!")
            return

        if not model_path or not os.path.exists(model_path):
            st.error("Please provide a valid Llama model path!")
            return

        # Save uploaded file temporarily
        with st.spinner("Processing document..."):
            try:
                # Create a temporary file
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Initialize Summarizer
                summarizer = DocumentSummarizer(
                    model_path=model_path,
                    context_length=context_length,
                    n_gpu_layers=-1
                )

                # Generate Summary
                summary = summarizer.process_document(uploaded_file.name)

                # Display Results
                if summary:
                    st.success("Summary Generated Successfully!")
                    st.markdown("### 📝 Summary:")
                    st.write(summary)
                else:
                    st.warning("Could not generate summary. Please check the document.")

                # Clean up temporary file
                os.remove(uploaded_file.name)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by HOPS*")

if __name__ == "__main__":
    main()