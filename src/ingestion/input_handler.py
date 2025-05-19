import os

from src.ingestion.text_loader import TextLoader


class InputHandler:
    def __init__(self, config=None):
        self.config = config or {}
        self.text_extensions = {'.txt'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '??????'}

    def detect_input_type(self, file_path=None, file_content=None, mime_type=None):
        """
        Detect whether the input is text or image based on extension, content, or mime type.
        Returns 'text', 'image', or 'unknown'
        """
        # Check by file extension if path is provided
        if file_path:
            extension = os.path.splitext(file_path)[1].lower()
            if extension in self.text_extensions:
                return 'text'
            elif extension in self.image_extensions:
                return 'image'

        if mime_type:
            if mime_type.startswith('text/'):
                return 'text'
            elif mime_type.startswith('image/'):
                return 'image'

        if file_content:
            # Try to detect if it's binary (image) or text
            try:
                # If decoding as UTF-8 works without errors, likely text
                file_content[:1024].decode('utf-8')
                return 'text'
            except (UnicodeDecodeError, AttributeError):
                # Check for common image file signatures/magic numbers
                if file_content.startswith(b'\xFF\xD8\xFF'):  # JPEG
                    return 'image'
                elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                    return 'image'
                elif file_content.startswith(b'GIF8'):  # GIF
                    return 'image'
                # Add other magic numbers as needed

        return 'unknown'

    def process_input(self, input_source):
        """
        Main method to handle input and route to appropriate pipeline
        """
        file_path = None
        file_content = None
        mime_type = None

        # Handle different input source types
        if isinstance(input_source, str):
            # Check if it's a file path or a URL
            if os.path.exists(input_source):
                file_path = input_source
                with open(file_path, 'rb') as f:
                    file_content = f.read()
            elif input_source.startswith(('http://', 'https://')):
                # Handle URL (you'd implement URL fetching here)
                pass
        elif isinstance(input_source, bytes):
            file_content = input_source

        # Detect input type
        input_type = self.detect_input_type(file_path, file_content, mime_type)

        # Route to appropriate pipeline
        if input_type == 'text':
            return self._route_to_text_pipeline(file_path, file_content, mime_type)
        # elif input_type == 'image':
        #     return self._route_to_image_pipeline(file_path, file_content, mime_type)
        else:
            raise ValueError("Unknown input type. Cannot determine if text or image.")

    def _route_to_text_pipeline(self, file_path, file_content, mime_type):
        """Route to text processing pipeline"""
        # Initialize text pipeline components
        text_loader = TextLoader()
        text_data = text_loader.load(file_path, file_content, mime_type)
        return {'type': 'text', 'data': text_data, 'path': file_path}


    # def _route_to_image_pipeline(self, file_path, file_content, mime_type):
    #     """Route to image processing pipeline"""
    #     # Initialize image pipeline components
    #     image_loader = ImageLoader()
    #     image_data = image_loader.load(file_path, file_content, mime_type)
    #     return {'type': 'image', 'data': image_data, 'path': file_path}