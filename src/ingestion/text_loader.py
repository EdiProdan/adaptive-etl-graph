import os


class TextLoader:
    def __init__(self):
        self.format_handlers = {
            '.txt': self._load_plain_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.html': self._load_html,
            # Add more format handlers as needed
        }

    def load(self, file_path=None, file_content=None, mime_type=None):
        """Load text content from the provided source"""
        if file_path:
            extension = os.path.splitext(file_path)[1].lower()
            handler = self.format_handlers.get(extension, self._load_plain_text)
            return handler(file_path, file_content)
        elif file_content:
            # Try to interpret as plain text if no path provided
            return self._decode_text_content(file_content)

        raise ValueError("No file path or content provided")

    def _load_plain_text(self, file_path, file_content=None):
        if file_content:
            return self._decode_text_content(file_content)
        with open(file_path, 'rb') as f:
            return self._decode_text_content(f.read())

    def _load_pdf(self, file_path, file_content=None):
        # Use PyPDF2, pdfplumber, or similar to extract text
        # Example implementation would go here
        pass

    def _load_docx(self, file_path, file_content=None):
        # Use python-docx or similar to extract text
        # Example implementation would go here
        pass

    def _load_html(self, file_path, file_content=None):
        # Use BeautifulSoup or similar to extract text
        # Example implementation would go here
        pass

    def _decode_text_content(self, content):
        """Try to decode binary content as text using various encodings"""
        encodings = ['utf-8', 'latin-1', 'utf-16', 'ascii']
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeError("Could not decode text content with any known encoding")