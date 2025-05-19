from sentence_transformers import SentenceTransformer

class SBERTEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)
