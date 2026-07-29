import re

def parse_query_intent(query: str) -> dict:
    """
    Analyzes a Roman Urdu / English query using Regex to determine the user's intent 
    and extract constraints (like price limits) without calling an LLM.
    """
    query_lower = query.lower().strip()
    
    # Initialize our return variables
    max_price = None
    clean_query = query_lower
    
    # ==========================================
    # 1. PRICE FILTER DETECTION
    # ==========================================
    
    # Pattern A: Prefix triggers (e.g., "under 500", "budget 1000", "max 2000 pkr")
    # Matches: (trigger words) + optional currency + (numbers)
    price_prefix_pattern = r'\b(under|max|budget of|less than)\s*(?:rs\.?|pkr|rupees)?\s*(\d+)\b'
    
    # Pattern B: Suffix triggers (e.g., "500 se kam", "1000 tak", "2000 ke andar")
    # Matches: (numbers) + optional currency + (trigger words)
    price_suffix_pattern = r'\b(\d+)\s*(?:rs\.?|pkr|rupees)?\s*(se kam|tak|k andar|ke andar)\b'
    
    # Check for Prefix matches
    prefix_match = re.search(price_prefix_pattern, clean_query)
    if prefix_match:
        max_price = int(prefix_match.group(2))
        # Remove the price phrase from the query so it doesn't pollute the vector search
        clean_query = re.sub(price_prefix_pattern, '', clean_query)
        
    # Check for Suffix matches (if prefix wasn't found)
    if not max_price:
        suffix_match = re.search(price_suffix_pattern, clean_query)
        if suffix_match:
            max_price = int(suffix_match.group(1))
            # Remove the price phrase from the query
            clean_query = re.sub(price_suffix_pattern, '', clean_query)

    # ==========================================
    # 2. RECIPE / MULTI-ITEM DETECTION
    # ==========================================
    
    # Matches English and Roman Urdu recipe keywords
    recipe_pattern = r'\b(recipe|ingredients for|how to make|how to cook|banan[ea]y? ka tarika|kaise banay[ea]n|ki recipe|banana hai)\b'
    
    is_recipe = bool(re.search(recipe_pattern, clean_query))
    if is_recipe:
        # We also remove the recipe keywords from the query to leave just the core subject (e.g., "biryani")
        clean_query = re.sub(recipe_pattern, '', clean_query)

    # Clean up any accidental double spaces left over from removing words
    clean_query = re.sub(r'\s+', ' ', clean_query).strip()
    
    # ==========================================
    # 3. DETERMINE FINAL INTENT
    # ==========================================
    
    intent = "standard_search"
    
    if is_recipe:
        intent = "recipe_builder"
    elif max_price is not None:
        intent = "price_filter"
        
    return {
        "original_query": query,
        "clean_query": clean_query,
        "intent": intent,
        "max_price": max_price
    }

# ==========================================
# TEST EXAMPLES (Run this file to see it work)
# ==========================================
if __name__ == "__main__":
    test_queries = [
        "best shampoo under 500",
        "biryani bananay ka tarika",
        "1000 pkr se kam wale pampers",
        "ingredients for chicken karahi",
        "surf excel 1kg", # Standard search
        "budget 2000 rs mein body wash"
    ]
    
    for q in test_queries:
        result = parse_query_intent(q)
        print(f"Original : '{result['original_query']}'")
        print(f"Intent   : {result['intent']}")
        print(f"Cleaned  : '{result['clean_query']}'")
        if result['max_price']:
            print(f"Max Price: {result['max_price']} PKR")
        print("-" * 40)