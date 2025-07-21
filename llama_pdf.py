import os
from typing import Optional
import pytesseract
from PIL import Image
import PyPDF2
from llama_cpp import Llama
from googletrans import Translator


class DocumentSummarizer:
    def __init__(self, 
                 model_path: str, 
                 context_length: int = 4096,
                 n_gpu_layers: int = -1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            print("Llama model initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Error initializing Llama model: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n" if page_text else ""
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extracts text from an image using Tesseract."""
        try:
            image = Image.open(image_path)
            text_to_translate = pytesseract.image_to_string(image)
            translator = Translator()
            text = translator.translate(text_to_translate, src='auto', dest='en').text
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"Error extracting text from image: {e}")

    def summarize_text(self, text: str, max_tokens: int = 500) -> str:
        """Generates a summary for the given text using Llama."""
        prompt = f"""### Instruction:
        Provide a concise and comprehensive summary of the following text. 
        Focus on the key points, main arguments, and essential information.
        Keep the summary clear, objective.

        ### Input:
        {text}

        ### Summary:
        """
        try:
            print(f"Prompt sent to Llama:\n{prompt[:500]}...")
            response = self.llm(
                prompt, 
                max_tokens=max_tokens,
                stop=["### Input:", "### Summary:"],
                echo=False
            )
            if 'choices' in response and response['choices']:
                summary = response['choices'][0]['text'].strip()
                print("Summary generated successfully.")
                return summary
            else:
                raise ValueError("Invalid response format from Llama.")
        except Exception as e:
            print(f"Error during summarization: {e}")
            return "Unable to generate summary."
    
    def process_document(self, file_path: str) -> Optional[str]:
        """Processes a document (PDF or image) and generates a summary."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                print(f"Processing PDF: {file_path}")
                text = self.extract_text_from_pdf(file_path)
            elif file_ext in ['.png', '.jpg', '.jpeg', '.webp']:
                print(f"Processing Image: {file_path}")
                text = self.extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            return self.summarize_text(text)
        
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return None


def main():
    MP = "D:/PROJECT EXPO 1/Harshgup16/llama-3-8b-Instruct-bnb-4bit-laptop-recommendation/unsloth.Q4_K_M.gguf"
    PP = "Sample-filled-in-MR.pdf"
    IP = "graph.jpg"

    try:
        summarizer = DocumentSummarizer(
            model_path=MP,
            context_length=4096,
            n_gpu_layers=-1
        )

        pdf_summary = summarizer.process_document(PP)
        if pdf_summary:
            print("PDF Summary:\n", pdf_summary)

        image_summary = summarizer.process_document(IP)
        if image_summary:
            print("Image Summary:\n", image_summary)
    
    except Exception as e:
        print(f"Initialization error: {e}")


if __name__ == "__main__":
    main()