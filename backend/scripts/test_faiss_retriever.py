from backend.retrieval.faiss_retriever import FAISSRetriever

r = FAISSRetriever()

results = r.search(
    "निगम या कंपनी की परिभाषा क्या है?",
    top_k=3,
)

for res in results:
    print(
        f"[{res['score']:.4f}] "
        f"{res['chunk_id']} -> "
        f"{res['text'][:100]}..."
    )