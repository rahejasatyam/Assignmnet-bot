"""Quick QA check on catalog.json"""
import json, sys
sys.path.insert(0, '.')

with open('data/catalog.json', encoding='utf-8') as f:
    d = json.load(f)

bad_urls = [a for a in d if not a['url'].startswith('https://www.shl.com/products/product-catalog/')]
has_desc = sum(1 for a in d if a.get('description', '').strip())
has_types = sum(1 for a in d if a.get('test_types'))

print(f"Total assessments : {len(d)}")
print(f"Bad URLs          : {len(bad_urls)}")
print(f"With description  : {has_desc}/{len(d)}")
print(f"With test types   : {has_types}/{len(d)}")

if bad_urls:
    print("\nBAD URLs:")
    for b in bad_urls[:5]:
        print(f"  {b['name']} -> {b['url']}")

print("\n── Sample entries ──")
for a in d[:5]:
    print(f"  {a['name']}")
    print(f"    url  : {a['url']}")
    print(f"    types: {a['test_types']}")
    print(f"    desc : {a.get('description','')[:80]}")
    print()
