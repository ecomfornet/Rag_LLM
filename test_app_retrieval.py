import os
import requests
import json

# App exposes /api/query (GET ?query=... or POST {"query": "..."})
url = "http://localhost:5000/api/query"
# GET with query param
query = "Arrêt N° 004 du 11 Janvier 2017"

try:
    response = requests.get(url, params={"query": query}, timeout=60)
    print("Status:", response.status_code)
    data = response.json()
    if data.get("error"):
        print("Error:", data["error"])
    else:
        print("Answer:", data.get('answer', '')[:500])
    
    print("\nSources:")
    for i, src in enumerate(data.get('sources', [])):
        print(f"  {i+1}. {src.get('title')} (Ref: {src.get('reference')})")
        
except Exception as e:
    print("Error:", e)
