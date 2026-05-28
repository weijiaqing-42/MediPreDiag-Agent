"""
Download 15 real PDFs from PubMed Central.
Step 1 only: search + download PDFs to disk. No text extraction.
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pubmed_pdfs"
MANIFEST_FILE = PDF_DIR / "manifest.json"

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"
RATE_LIMIT = 0.5

SEARCH_TOPICS = [
    ("糖尿病", "diabetes"),
    ("高血压", "hypertension"),
    ("癌症免疫治疗", "cancer immunotherapy"),
    ("抗生素耐药", "antibiotic resistance"),
    ("阿尔茨海默病", "Alzheimer disease"),
    ("心力衰竭", "heart failure"),
    ("哮喘", "asthma"),
    ("脑卒中", "stroke"),
    ("抑郁症", "depression"),
    ("肥胖症", "obesity"),
    ("慢性肾病", "chronic kidney disease"),
    ("乙型肝炎", "hepatitis B"),
    ("结核病", "tuberculosis"),
    ("类风湿关节炎", "rheumatoid arthritis"),
    ("COVID-19", "COVID-19"),
]

HEADERS = {"User-Agent": "MediPreDiag/1.0 (research@example.com)"}


def log(msg: str):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def esearch(query: str, retmax: int = 5) -> list[str]:
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        params = {
            "db": "pmc",
            "term": f'{query} AND open access[filter]',
            "retmax": retmax, "retmode": "json", "sort": "relevance",
        }
        resp = await client.get(f"{NCBI_BASE}/esearch.fcgi", params=params)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])


async def esummary(pmcids: list[str]) -> list[dict]:
    if not pmcids:
        return []
    async with httpx.AsyncClient(headers=HEADERS, timeout=20) as client:
        params = {"db": "pmc", "id": ",".join(pmcids), "retmode": "json"}
        resp = await client.get(f"{NCBI_BASE}/esummary.fcgi", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for pid in pmcids:
            item = data.get("result", {}).get(pid)
            if item:
                results.append({"pmcid": pid, "title": item.get("title", ""),
                                "pubdate": item.get("pubdate", "")})
        return results


async def find_pdf_link(pmcid: str) -> str | None:
    """Fetch PMC article page and extract the PDF download URL."""
    try:
        async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
                                     follow_redirects=True) as client:
            resp = await client.get(f"{PMC_BASE}/{pmcid}/")
            resp.raise_for_status()
            html = resp.text
            matches = re.findall(r'href=["\']([^"\']*\.pdf[^"\']*)["\']', html, re.IGNORECASE)
            for m in matches:
                if not m.endswith("supplement") and "supplement" not in m.lower() and "suppl" not in m.lower():
                    pdf_url = m
                    if pdf_url.startswith("pdf/"):
                        return f"{PMC_BASE}/{pmcid}/{pdf_url}"
                    if pdf_url.startswith("/"):
                        return f"https://www.ncbi.nlm.nih.gov{pdf_url}"
                    return pdf_url
            return None
    except Exception as e:
        log(f"    find_pdf_link failed: {e}")
        return None


def download_pdf_streaming(url: str, filepath: Path) -> int:
    """Stream PDF to disk in 256KB chunks. Returns file size."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        ct = resp.headers.get("Content-Type", "")
        if "html" in ct.lower():
            log(f"    Got HTML instead of PDF")
            return 0
        with open(filepath, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return total


async def main():
    print("=" * 60, flush=True)
    print("PMC PDF Downloader - Step 1: Download only", flush=True)
    print("=" * 60, flush=True)

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, (cn_name, en_query) in enumerate(SEARCH_TOPICS):
        print(f"\n[{i+1}/{len(SEARCH_TOPICS)}] {cn_name}", flush=True)

        pmcids = await esearch(en_query)
        if not pmcids:
            log(f"no results, skip")
            continue

        summaries = await esummary(pmcids)
        await asyncio.sleep(RATE_LIMIT)

        downloaded = False
        for s in summaries:
            pmcid = s["pmcid"]
            title = s["title"]

            log(f"  Trying {pmcid}: {title[:80]}...")
            pdf_url = await find_pdf_link(pmcid)
            if not pdf_url:
                log(f"    no PDF link found")
                continue

            log(f"    Downloading: {pdf_url}")
            filepath = PDF_DIR / f"{pmcid}.pdf"
            try:
                size = download_pdf_streaming(pdf_url, filepath)
                if size < 5000:
                    filepath.unlink(missing_ok=True)
                    log(f"    too small ({size}B), try next")
                    continue
                log(f"    OK: {filepath.name} ({size}B)")
                manifest.append({
                    "pmcid": pmcid, "title": title,
                    "topic_cn": cn_name, "topic_en": en_query,
                    "pubdate": s.get("pubdate", ""),
                    "file": str(filepath.name), "size": size,
                })
                downloaded = True
                break
            except Exception as e:
                log(f"    download error: {e}")
                filepath.unlink(missing_ok=True)
                continue

        if not downloaded:
            log(f"  *** FAILED to get PDF for {cn_name} ***")
        await asyncio.sleep(RATE_LIMIT * 2)

    print(f"\n{'='*60}", flush=True)
    print(f"Downloaded {len(manifest)}/{len(SEARCH_TOPICS)} PDFs", flush=True)
    for m in manifest:
        print(f"  {m['pmcid']}: {m['topic_cn']} - {m['size']}B", flush=True)

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest saved: {MANIFEST_FILE}", flush=True)
    print(f"PDFs saved in: {PDF_DIR}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())