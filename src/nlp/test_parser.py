from src.nlp.query_parser import parse_query

query = "I’m planning a trip to Denver. What can I see and do there?"
parsed = parse_query(query)

print("Parsed info:", parsed)