from retrieval_4 import HybridRetriever
import json

retriever = HybridRetriever()
query = "Arrêt N° 004 du 11 Janvier 2017"

print("--- Query Analysis ---")
plan = retriever.analyze_query(query)
print(json.dumps(plan, indent=2, ensure_ascii=False))
