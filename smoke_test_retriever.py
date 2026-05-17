import sys
sys.path.insert(0, '.')
from retriever.retriever import search, get_all_urls

results = search('Java developer cognitive ability', k=5)
print('--- Search: Java developer ---')
for r in results:
    score = r['_score']
    name = r['name']
    types = r['test_types']
    url = r['url']
    print(f'  [{score:.3f}] {name} | types={types}')
    print(f'           {url}')

all_urls = get_all_urls()
print(f'\nTotal catalog URLs in index: {len(all_urls)}')

results2 = search('personality assessment leadership sales', k=3)
print('\n--- Search: personality leadership ---')
for r in results2:
    score = r['_score']
    name = r['name']
    types = r['test_types']
    print(f'  [{score:.3f}] {name} | types={types}')

results3 = search('customer service call center simulation', k=3)
print('\n--- Search: customer service ---')
for r in results3:
    score = r['_score']
    name = r['name']
    types = r['test_types']
    print(f'  [{score:.3f}] {name} | types={types}')
