\
import os

class ContentExtractor:
    """
    Extracts raw text content from supported plain text files.
    """
    def __init__(self):
        # Common plain text and code file extensions
        self.supported_extensions = [
            '.txt', '.md', '.py', '.java', '.html', '.sh', '.js', '.css',
            '.xml', '.json', '.csv', '.log', '.rst', '.tex', '.c', '.cpp',
            '.h', '.hpp', '.go', '.rb', '.php', '.swift', '.kt', '.scala'
        ]

    def extract_text(self, file_path: str) -> str | None:
        """
        Extracts raw text content from the given file if its type is supported.

        Args:
            file_path: The absolute path to the file.

        Returns:
            The extracted text content, or None if the file is not
            supported or an error occurs during reading.
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() in self.supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text from {file_path}: {e}")
                return None
        else:
            # This can be noisy if many unsupported files are scanned.
            # print(f"Unsupported file type for text extraction: {file_path}")
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
