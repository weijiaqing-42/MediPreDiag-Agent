"""Quick test: find PDF download links on PMC article pages."""
import urllib.request
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}

pmcids = ["PMC9811334", "PMC3163656", "PMC4678005", "PMC8112876", "PMC9912345"]

for pmcid in pmcids:
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=20)
        html = resp.read().decode("utf-8", errors="replace")
        pdf_links = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE)
        print(f"{pmcid}: {len(html)} bytes, PDF links: {pdf_links}")
        print(f"  Title tag: {re.findall(r'<title>(.*?)</title>', html)}")
    except Exception as e:
        print(f"{pmcid}: ERROR - {e}")