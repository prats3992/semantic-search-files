import os
from docx import Document
from typing import Optional

GEMINI_MODEL_NAME = "gemini-2.5-flash"


class ContentExtractor:
    """
    Extracts raw text content from supported plain text files, uses specific
    handlers for formats like .docx, and can leverage Gemini for PDFs and images.
    """
    def __init__(self, gemini_api_key: Optional[str] = None):
        # Extensions for direct plain text reading
        self.plain_text_extensions = [
            '.txt', '.md', '.py', '.java', '.html', '.sh', '.js', '.css',
            '.xml', '.json', '.csv', '.log', '.rst', '.tex', '.c', '.cpp',
            '.h', '.hpp', '.go', '.rb', '.php', '.swift', '.kt', '.scala'
        ]
        # Only these are ever uploaded to Gemini, so arbitrary binaries are never sent off-machine.
        self.gemini_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.webp']

        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._gemini_client = None

    def extract_text(self, file_path: str) -> str | None:
        """
        Extracts raw text content from the given file if its type is supported.
        Uses python-docx for .docx files and Gemini (when an API key is set) for PDFs and images.

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
        if file_extension in self.plain_text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading plain text from {file_path}: {e}")
                return None
        if file_extension in self.gemini_extensions:
            if self.gemini_api_key:
                return self._extract_text_with_gemini(file_path)
            print(f"Skipping {file_path}: set GEMINI_API_KEY to extract text from {file_extension} files.")
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

    def _extract_text_with_gemini(self, file_path: str) -> Optional[str]:
        """Extracts text from a PDF or image by uploading it to the Gemini API."""
        print(f"Extracting text from {file_path} using Gemini ({GEMINI_MODEL_NAME})...")
        try:
            if self._gemini_client is None:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
            client = self._gemini_client

            uploaded_file = client.files.upload(file=file_path)
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[
                        "Extract all text from this file. If it is an image, perform OCR. "
                        "Return only the extracted text.",
                        uploaded_file,
                    ],
                )
            finally:
                client.files.delete(name=uploaded_file.name)

            if response and response.text:
                return response.text
            print(f"Gemini could not extract text from {file_path}.")
            return None
        except Exception as e:
            print(f"Error extracting text with Gemini from {file_path}: {e}")
            return None
