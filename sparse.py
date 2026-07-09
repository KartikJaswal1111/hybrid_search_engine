"""TF-IDF sparse retriever — ranks documents by lexical/token overlap with the query."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "documents.json"


def load_documents():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


class SparseRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([doc["text"] for doc in documents])

    def search(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    docs = load_documents()
    retriever = SparseRetriever(docs)
    query = "What does card FUT-23-091 belong to?"
    for doc, score in retriever.search(query, top_k=3):
        print(f"[{score:.3f}] id={doc['id']}  {doc['text']}")
