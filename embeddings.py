import numpy as np

from config import EMBEDDING_MODEL


def embed_text(client, text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def vector_to_blob(vector):
    return vector.astype(np.float32).tobytes()


def blob_to_vector(blob):
    return np.frombuffer(blob, dtype=np.float32)
