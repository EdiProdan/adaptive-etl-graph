from src.preprocessing.text.cleaner import TextCleaner
from src.preprocessing.text.segmenter import TextSegmenter


class TextPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.text_cleaner = TextCleaner()
        self.text_segmenter = TextSegmenter()

    def process(self, text_data):
        """
        Process text data through the full text pipeline
        """

        cleaned_text = self.text_cleaner.clean(text_data['data'])

        segmented_text = self.text_segmenter.segment_text(cleaned_text)

        # Print results
        print(f"Number of paragraphs: {len(segmented_text['paragraphs'])}")
        print(f"Number of sentences: {len(segmented_text['sentences'])}")
        print("\nSentences:")
        for i, sentence in enumerate(segmented_text['sentences']):
            print(f"  {i + 1}. {sentence}")

        # Get sentences with entities
        sentences_with_entities = self.text_segmenter.get_sentence_entities(cleaned_text)

        print("\nSentences with entities:")
        for i, sent_data in enumerate(sentences_with_entities):
            print(f"  {i + 1}. {sent_data['text']}")
            for ent in sent_data['entities']:
                print(f"     - {ent['text']} ({ent['label']})")
        return {
            'type': 'text',
            'cleaned_text': cleaned_text
        }
