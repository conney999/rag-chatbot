from sentence_transformers import SentenceTransformer
from config import ALL_MINILM_MODEL_NAME, BGE_MODEL_NAME, MPNET_MODEL_NAME, CACHED_DIR

class BaseSTEmbedder:
    def __init__(self, model_name, instruction=None):
        self.model = SentenceTransformer(model_name, cache_folder=CACHED_DIR)
        self.instruction = instruction

    def encode(self, text, query=False):
        if query and self.instruction:
            text = [self.instruction + t for t in text] 
        return self.model.encode(text, normalize_embeddings=True)


class AllMiniLM_Model(BaseSTEmbedder):
    def __init__(self):
        super().__init__(ALL_MINILM_MODEL_NAME)


class Bge_Model(BaseSTEmbedder):
    def __init__(self):
        super().__init__(
            model_name=BGE_MODEL_NAME, instruction="Represent this sentence for searching relevant passages: "
        )

class Mpnet_Model(BaseSTEmbedder):
    def __init__(self):
        super().__init__(MPNET_MODEL_NAME)

    