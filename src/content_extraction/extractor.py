\
import os
from docx import Document
import google.generativeai as genai
from typing import Optional

class ContentExtractor:
    """
    Extracts raw text content from supported plain text files, uses specific
    handlers for formats like .docx, and can leverage Gemini for others.
    """
    def __init__(self, gemini_api_key: Optional[str] = None):
        # Extensions for direct plain text reading
        self.plain_text_extensions = [
            '.txt', '.md', '.py', '.java', '.html', '.sh', '.js', '.css',
            '.xml', '.json', '.csv', '.log', '.rst', '.tex', '.c', '.cpp',
            '.h', '.hpp', '.go', '.rb', '.php', '.swift', '.kt', '.scala'
        ]
        # All potentially processable extensions (plain text + specific handlers + Gemini candidates)
        # This list can be expanded as more specific handlers or Gemini-supported types are confirmed.
        self.all_processable_extensions = self.plain_text_extensions + ['.docx'] # Add other known types here

        self.gemini_api_key = gemini_api_key
        if self.gemini_api_key:
            self._configure_gemini(self.gemini_api_key)

    def extract_text(self, file_path: str) -> str | None:
        """
        Extracts raw text content from the given file if its type is supported.
        Uses python-docx for .docx files and Gemini for other specified types.

        Args:
            file_path: The absolute path to the file.

        Returns:
            The extracted text content, or None if the file is not
            supported or an error occurs during reading.
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        if file_extension == '.docx':
            return self._extract_docx_text(file_path)
        elif file_extension in self.plain_text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading plain text from {file_path}: {e}")
                return None
        else:
            # Fallback for other file types: attempt with Gemini if API key is available
            if self.gemini_api_key:
                print(f"File type {file_extension} not directly handled, attempting with Gemini for {file_path}")
                return self._extract_text_with_gemini(file_path, self.gemini_api_key)
            else:
                print(f"File type {file_extension} not handled. Gemini API key not configured. Skipping {file_path}")
                return None

    def _extract_docx_text(self, file_path: str) -> Optional[str]:
        """Extracts text from a .docx file."""
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Error extracting DOCX text from {file_path}: {e}")
            return None

    def _configure_gemini(self, api_key: str):
        """Configures the Gemini API."""
        if api_key:
            genai.configure(api_key=api_key)
        else:
            print("Warning: Gemini API key not provided. Gemini functionality will be disabled.")

    def _extract_text_with_gemini(self, file_path: str, api_key_from_config: Optional[str] = None) -> Optional[str]:
        """
        Extracts text from a file using the Gemini API.
        Requires the API key to be configured.
        """
        if not api_key_from_config:
            print("Error: Gemini API key not available. Cannot extract text with Gemini.")
            return None
        
        self._configure_gemini(api_key_from_config)
        
        try:
            # Check if model is available (basic check)
            model_info = genai.get_model("models/gemini-2.5-flash-preview") # Corrected model name
            if not model_info: # Simplified check, actual availability might need a call
                 print(f"Error: Gemini model 'gemini-2.5-flash-preview' not found or accessible.")
                 return None

            model = genai.GenerativeModel('gemini-2.5-flash-preview') # Corrected model name

            print(f"Attempting to extract text from {file_path} using Gemini...")
            # Upload the file to Gemini (this is a conceptual step, actual API might differ)
            # For file reading, Gemini usually takes bytes or a file object directly in the request.
            # The `genai.upload_file` is for persistent files, which might not be what we want for quick extraction.
            # Let's assume we read the file and pass its content.
            # However, for robust file handling, the API might have a specific way to upload.
            # For this iteration, we'll use a simplified approach if the file is small enough,
            # otherwise, a file upload mechanism would be needed.

            # This is a placeholder for how you might use the file.
            # The actual Gemini API for file input in `generate_content` might vary.
            # It's common to pass `Part.from_uri` or `Part.from_data`.
            # Let's simulate reading the file and passing its content if it's text-based,
            # or using file upload for binary types.

            # For a general solution, using the file upload API is better.
            uploaded_file = genai.upload_file(path=file_path)
            
            response = model.generate_content([
                "Please extract all text from this file. If it's an image, perform OCR. If it's a document, extract all textual content.",
                uploaded_file
            ])
            
            # Clean up the uploaded file after use
            genai.delete_file(uploaded_file.name)

            if response and response.text:
                return response.text
            else:
                print(f"Gemini could not extract text from {file_path}. Response: {response}")
                return None
        except Exception as e:
            print(f"Error extracting text with Gemini from {file_path}: {e}")
            return None

if __name__ == '__main__':
    extractor = ContentExtractor()
    # Example: Create a dummy file for testing
    # test_file_path = "temp_extract_test.txt"
    # with open(test_file_path, "w") as f:
    #     f.write("This is a test file for content extraction.\\nIt has multiple lines.")
    #
    # text = extractor.extract_text(test_file_path)
    # if text:
    #     print(f"Extracted text from {test_file_path}:\\n{text}")
    # else:
    #     print(f"Could not extract text from {test_file_path}.")
    # if os.path.exists(test_file_path):
    #     os.remove(test_file_path)
    #
    # # Example: Test with its own source file
    # text_py = extractor.extract_text(__file__)
    # if text_py:
    #     print(f"\\nExtracted text from {__file__}:\\nFirst 100 chars: {text_py[:100].replace('\\n', ' ')}...")
    pass
