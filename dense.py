"""Dense embedding retriever — ranks documents by semantic similarity with the query."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sparse import load_documents

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(self, documents, model_name=MODEL_NAME):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([doc["text"] for doc in documents])

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query])
        scores = cosine_similarity(query_emb, self.embeddings)[0]
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    docs = load_documents()
    retriever = DenseRetriever(docs)
    query = "Need a poacher who never misses in the box"
    for doc, score in retriever.search(query, top_k=3):
        print(f"[{score:.3f}] id={doc['id']}  {doc['text']}")
