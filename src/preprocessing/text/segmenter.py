import spacy
from typing import Dict, List, Any, Optional, Tuple
import re


class TextSegmenter:
    """
    A text segmenter that uses spaCy to segment text into paragraphs, sentences, and other structural units.
    """

    def __init__(self, model: str = "en_core_web_md", custom_sentence_boundaries: bool = False):
        """
        Initialize the text segmenter with the specified spaCy model.

        Args:
            model: The spaCy model to use
            custom_sentence_boundaries: Whether to use custom sentence boundary detection rules
        """
        self.nlp = spacy.load(model)

    def segment_text(self, text: str) -> Dict[str, Any]:
        """
        Segment the input text into paragraphs, sentences, and structural units.

        Args:
            text: The input text to segment

        Returns:
            Dictionary containing segmented text units
        """
        # Split text into paragraphs (split by double newlines)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        all_sentences = []
        all_sentence_spans = []
        paragraph_to_sentences = {}

        # Process each paragraph
        for i, paragraph in enumerate(paragraphs):
            # Process through spaCy
            doc = self.nlp(paragraph)

            # Extract sentences from this paragraph
            paragraph_sentences = [sent.text.strip() for sent in doc.sents]

            # Get sentence spans relative to the paragraph
            paragraph_sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]

            # Calculate the starting index for sentences in this paragraph
            sentence_start_idx = len(all_sentences)

            # Add sentences to overall list
            all_sentences.extend(paragraph_sentences)
            all_sentence_spans.extend(paragraph_sentence_spans)

            # Map paragraph index to sentence indices
            sentence_indices = list(range(sentence_start_idx, sentence_start_idx + len(paragraph_sentences)))
            paragraph_to_sentences[i] = sentence_indices

        # Try to identify structural segments like sections, subsections, etc.
        structural_segments = self._identify_structural_segments(paragraphs)

        # Construct the final result
        result = {
            'paragraphs': paragraphs,
            'sentences': all_sentences,
            'sentence_spans': all_sentence_spans,
            'paragraph_to_sentences': paragraph_to_sentences,
            'structural_segments': structural_segments
        }

        return result

    def _identify_structural_segments(self, paragraphs: List[str]) -> Dict[str, List[Any]]:
        """
        Identify structural segments like sections, subsections, lists, etc.
        This is a basic implementation - you might want to extend it for your specific needs.

        Args:
            paragraphs: List of paragraphs

        Returns:
            Dictionary with identified structural segments
        """
        # Initialize result
        structural_segments = {
            'sections': [],
            'subsections': [],
            'lists': []
        }

        # Look for potential section headers
        section_pattern = re.compile(r'^(?:\d+\.\s+)?([A-Z][^.!?]+)$')

        # Look for bullet or numbered lists
        list_item_pattern = re.compile(r'^(?:\d+\.|\*|\-|\•)\s+')
        current_list = []

        for i, paragraph in enumerate(paragraphs):
            # Check if this paragraph looks like a section header
            if section_pattern.match(paragraph):
                # Simple heuristic: if it's short and ends with no punctuation, it might be a header
                if len(paragraph.split()) <= 10:
                    structural_segments['sections'].append({
                        'index': i,
                        'text': paragraph,
                        'level': 1  # Assume top-level section
                    })

            # Check if this paragraph is a list item
            if list_item_pattern.match(paragraph):
                current_list.append(i)
            elif current_list:
                # If we've been building a list and hit a non-list paragraph, end the list
                if len(current_list) > 1:  # Only consider it a list if there are multiple items
                    structural_segments['lists'].append(current_list.copy())
                current_list = []

        # Add any remaining list
        if len(current_list) > 1:
            structural_segments['lists'].append(current_list)

        return structural_segments

    def get_sentence_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Get sentences with their named entities.

        Args:
            text: Input text

        Returns:
            List of sentences with their entities:
            [
                {
                    'text': 'sentence text',
                    'entities': [{'text': 'entity text', 'label': 'entity label', 'start': start_char, 'end': end_char}, ...]
                },
                ...
            ]
        """
        # Process text with spaCy
        doc = self.nlp(text)

        sentences_with_entities = []

        # Iterate through sentences
        for sent in doc.sents:
            # Extract entities in this sentence
            entities = []
            for ent in doc.ents:
                # Check if entity belongs to this sentence
                if ent.start >= sent.start and ent.end <= sent.end:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char - sent.start_char,  # Start relative to sentence
                        'end': ent.end_char - sent.start_char  # End relative to sentence
                    })

            sentences_with_entities.append({
                'text': sent.text,
                'entities': entities
            })

        return sentences_with_entities

    def segment_with_context(self, text: str, window_size: int = 2) -> List[Dict[str, Any]]:
        """
        Segment text into sentences with surrounding context.

        Args:
            text: Input text
            window_size: Number of sentences to include before and after each sentence

        Returns:
            List of sentences with context:
            [
                {
                    'sentence': 'current sentence',
                    'context_before': ['previous sentence 1', 'previous sentence 2', ...],
                    'context_after': ['next sentence 1', 'next sentence 2', ...],
                    'position': {
                        'paragraph': paragraph_index,
                        'sentence': sentence_index
                    }
                },
                ...
            ]
        """
        # First segment the text
        segmented = self.segment_text(text)
        sentences = segmented['sentences']
        paragraph_to_sentences = segmented['paragraph_to_sentences']

        # Create a mapping from sentence index to paragraph index
        sentence_to_paragraph = {}
        for para_idx, sent_indices in paragraph_to_sentences.items():
            for sent_idx in sent_indices:
                sentence_to_paragraph[sent_idx] = para_idx

        # Build sentences with context
        sentences_with_context = []

        for i, sentence in enumerate(sentences):
            # Get context before
            context_before = sentences[max(0, i - window_size):i]

            # Get context after
            context_after = sentences[i + 1:min(len(sentences), i + 1 + window_size)]

            # Add to result
            sentences_with_context.append({
                'sentence': sentence,
                'context_before': context_before,
                'context_after': context_after,
                'position': {
                    'paragraph': sentence_to_paragraph.get(i, -1),
                    'sentence': i
                }
            })

        return sentences_with_context
