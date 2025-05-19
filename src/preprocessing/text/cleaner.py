"""
Text cleaner module for preprocessing plain text files.
"""
import re
import unicodedata
import html


class TextCleaner:
    """
    Cleans raw text by removing unwanted elements and standardizing formatting.
    Specifically optimized for TXT files.
    """

    def __init__(self, config=None):
        """
        Initialize the text cleaner with configuration options.

        Args:
            config (dict, optional): Configuration parameters for cleaning.
        """
        self.config = config or {}

        # Default configuration
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.remove_html = self.config.get('remove_html', True)
        self.normalize_whitespace = self.config.get('normalize_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.remove_brackets = self.config.get('remove_brackets', True)
        self.max_consecutive_newlines = self.config.get('max_consecutive_newlines', 2)

    def clean(self, text):
        """
        Clean the input text by applying various cleaning operations.

        Args:
            text (str): The raw text to clean.

        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""

        # HTML entity decoding (for TXT files that might have HTML entities)
        if self.remove_html:
            text = html.unescape(text)
            text = re.sub(r'<[^>]+>', '', text)

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+\.\S+', '', text)

        # Remove content in brackets if configured
        if self.remove_brackets:
            text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)

        # Normalize Unicode characters
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Normalize whitespace
        if self.normalize_whitespace:
            # Replace multiple spaces with a single space
            text = re.sub(r' +', ' ', text)

            # Limit consecutive newlines
            text = re.sub(r'\n{3,}', '\n' * self.max_consecutive_newlines, text)

            # Remove whitespace at the beginning and end of each line
            text = '\n'.join(line.strip() for line in text.split('\n'))

            # Remove leading and trailing whitespace
            text = text.strip()

        return text

    def clean_file(self, file_path):
        """
        Clean text content from a file.

        Args:
            file_path (str): Path to the text file to clean.

        Returns:
            str: The cleaned text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.clean(text)
        except UnicodeDecodeError:
            # Fall back to latin-1 if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            return self.clean(text)
        except Exception as e:
            raise IOError(f"Error reading or cleaning file {file_path}: {str(e)}")
