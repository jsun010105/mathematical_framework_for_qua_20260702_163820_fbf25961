#!/usr/bin/env python3
"""Query arXiv API and print relevance-sorted results (with retry/backoff)."""
import sys, time, urllib.parse, requests, xml.etree.ElementTree as ET, re

NS = {'a': 'http://www.w3.org/2005/Atom'}
HDR = {'User-Agent': 'Mozilla/5.0 ResearchBot (chicagohailab@gmail.com)'}

def search(query, max_results=8):
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({
        'search_query': query, 'start': 0, 'max_results': max_results,
        'sortBy': 'relevance', 'sortOrder': 'descending'})
    for attempt in range(5):
        r = requests.get(url, headers=HDR, timeout=60)
        if r.status_code == 200:
            break
        time.sleep(4 * (attempt + 1))
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall('a:entry', NS):
        aid = e.find('a:id', NS).text.strip()
        m = re.search(r'arxiv\.org/abs/(.+?)(v\d+)?$', aid)
        arxiv_id = m.group(1) if m else aid
        title = ' '.join(e.find('a:title', NS).text.split())
        summ = ' '.join(e.find('a:summary', NS).text.split())
        authors = [a.find('a:name', NS).text for a in e.findall('a:author', NS)]
        pub = e.find('a:published', NS).text[:4]
        out.append({'id': arxiv_id, 'title': title, 'year': pub,
                    'authors': ', '.join(authors[:4]), 'abstract': summ})
    return out

if __name__ == '__main__':
    q = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for i, p in enumerate(search(q, n), 1):
        print(f"\n[{i}] {p['id']} ({p['year']}) — {p['title']}")
        print(f"    Authors: {p['authors']}")
        print(f"    {p['abstract'][:380]}...")
