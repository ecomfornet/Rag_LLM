from retrieval_4 import HybridRetriever
import json

retriever = HybridRetriever()

# Simulate what analyze_query extracts
query = "Arrêt N° 004 du 11 Janvier 2017"

print(f"\n--- Testing analyze_query ---")
plan = retriever.analyze_query(query)
print(json.dumps(plan, ensure_ascii=False, indent=2))
