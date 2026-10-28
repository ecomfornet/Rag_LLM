import requests
import json
import time
import sys
import unicodedata

# Color formatting
class Colors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

API_URL = "http://localhost:5000/api/query"

TEST_CASES = [
    {
        "id": "TC01_JURIS_DETECT",
        "type": "Complex Jurisprudence",
        "query": "Selon l'arrêt de la Cour d'Appel de Kankan, Chambre du Premier Président, Arrêt N° 004 du 11 Janvier 2017, quels sont les éléments retenus pour constater le droit de propriété ?",
        "expected_route": "Complex/Full Pipeline",
        "expected_keywords": ["titre foncier", "donation", "décret", "D/94/180"],
        "min_chunks": 2,
        "min_confidence": 0.8
    },
    {
        "id": "TC02_SIMPLE_DEF",
        "type": "Simple Definition",
        "query": "Qu'est ce que la Cour Commune de Justice et d'Arbitrage (CCJA) ?",
        "expected_route": "SimpleFastPath",  # Or complex if the LLM decides so, but ideally simple
        "expected_keywords": ["arbitrage", "justice", "OHADA"],
        "min_chunks": 1,
        "min_confidence": 0.7
    },
    {
        "id": "TC03_TEMPORAL_SEARCH",
        "type": "Temporal & Entity",
        "query": "Quelles sont les décisions rendues en 2021 par la CCJA concernant les pourvois en cassation ?",
        "expected_route": "Complex/Full Pipeline",
        "expected_keywords": ["2021", "pourvoi", "cassation"],
        "min_chunks": 2,
        "min_confidence": 0.7
    }
]

def normalize_text(text):
    if not text: return ""
    return "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn').lower()

def clear_cache():
    try:
        requests.post("http://localhost:5000/api/clear", timeout=5)
    except:
        pass

def run_test(test_case):
    print(f"\n{Colors.HEADER}============================================================{Colors.ENDC}")
    print(f"{Colors.BOLD}Running: {test_case['id']} - {test_case['type']}{Colors.ENDC}")
    print(f"Query: {test_case['query']}")
    
    # Ensure fresh run
    clear_cache()
    
    payload = {"query": test_case['query'], "disable_cache": True}
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"{Colors.FAIL}Error calling API: {e}{Colors.ENDC}")
        return False, {}

    elapsed = time.time() - start_time
    
    # Parsing the response
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    trace = data.get("agent_trace", [])
    
    # Analyze trace for route
    route_taken = "Unknown"
    confidence = 0.0
    for step in trace:
        step_str = str(step)
        if "confidence=" in step_str:
            try:
                confidence = float(step_str.split("confidence=")[1].split()[0].strip().strip(","))
            except:
                pass
        if "RouterAgent" in step_str:
            if "SimpleFastPath" in str(trace):
                route_taken = "SimpleFastPath"
            else:
                route_taken = "Complex/Full Pipeline"
                
    # Run assertions
    success = True
    errors = []

    # 1. Check Route
    if route_taken != test_case["expected_route"] and test_case["expected_route"] != "ANY":
        if test_case["id"] == "TC01_JURIS_DETECT" and route_taken != "Complex/Full Pipeline":
             success = False
             errors.append(f"Route mismatch: Expected {test_case['expected_route']}, got {route_taken}")

    # 2. Check Retrieval Power (Source counts)
    num_sources = len(sources)
    if num_sources < test_case["min_chunks"]:
        success = False
        errors.append(f"Retrieval weak: Only {num_sources} unique sources retrieved, expected >= {test_case['min_chunks']}")
        
    # 3. Check Generation Power (Fact Checking / Confidence)
    if confidence < test_case["min_confidence"]:
        success = False
        errors.append(f"Confidence too low: {confidence}, expected >= {test_case['min_confidence']}")
        
    # 4. Content Power (Keyword recall - Accent Insensitive)
    normalized_answer = normalize_text(answer)
    missing_keywords = []
    for kw in test_case["expected_keywords"]:
        if normalize_text(kw) not in normalized_answer:
            missing_keywords.append(kw)
            
    if missing_keywords:
        success = False
        errors.append(f"Missing generation keywords: {missing_keywords}")

    # Output Results
    if success:
        print(f"{Colors.OKGREEN}[+] PASS {Colors.ENDC} ({elapsed:.2f}s) | Route: {route_taken} | Sources: {num_sources} | Conf: {confidence}")
    else:
        print(f"{Colors.FAIL}[-] FAIL {Colors.ENDC} ({elapsed:.2f}s) | Route: {route_taken} | Sources: {num_sources} | Conf: {confidence}")
        for err in errors:
            print(f"    -> {err}")
        if answer:
            print(f"\nAnswer Preview (First 300 chars):\n{answer[:300]}...")
            
    return success, {
        "time": elapsed,
        "route": route_taken,
        "chunks": num_sources,
        "confidence": confidence
    }

def main():
    print(f"{Colors.BOLD}Starting Comprehensive GraphRAG Evaluation Suite{Colors.ENDC}")
    print("Testing Agent Routing, Retrieval Expansion, and Generation Accuracy...\n")
    
    results = []
    passed = 0
    
    for i, test in enumerate(TEST_CASES):
        if i > 0:
             time.sleep(2)
        success, metrics = run_test(test)
        results.append((test["id"], success, metrics))
        if success:
            passed += 1

    print(f"\n{Colors.HEADER}============================================================{Colors.ENDC}")
    print(f"{Colors.BOLD}EVALUATION SUMMARY{Colors.ENDC}")
    print(f"Total Tests : {len(TEST_CASES)}")
    print(f"Passed      : {Colors.OKGREEN}{passed}{Colors.ENDC}")
    print(f"Failed      : {Colors.FAIL}{len(TEST_CASES) - passed}{Colors.ENDC}")
    
    print("\nMetrics table:")
    print(f"{'Test ID':<20} | {'Status':<6} | {'Route':<22} | {'Sources':<6} | {'Conf':<4} | {'Time(s)':<6}")
    print("-" * 75)
    for t_id, success, m in results:
        status = "PASS" if success else "FAIL"
        r = m.get('route', 'N/A')
        c = m.get('chunks', 0)
        cnf = m.get('confidence', 0.0)
        t = m.get('time', 0.0)
        print(f"{t_id:<20} | {status:<6} | {r:<22} | {c:<6} | {cnf:.2f} | {t:.2f}")

if __name__ == "__main__":
    main()
