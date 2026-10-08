"""
URL Article Extraction Module.
Uses trafilatura with a BeautifulSoup fallback to reliably extract
the headline and main body text from online news article URLs.
Includes specialized handling for bot-protected and paywalled domains.
"""

import re
import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    "Connection": "keep-alive"
}

# Known sites that deploy strict commercial anti-bot WAFs (Akamai, Cloudflare, PerimeterX)
BOT_PROTECTED_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "nytimes.com",
    "washingtonpost.com"
]

def is_bot_protected_url(url: str) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in BOT_PROTECTED_DOMAINS)

def extract_article_from_url(url: str, timeout: int = 10) -> dict:
    """
    Downloads and extracts article title and body text from a given URL.
    Returns:
    {
        "success": bool,
        "title": str,
        "text": str,
        "url": str,
        "error": str or None,
        "is_bot_blocked": bool
    }
    """
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    title = ""
    text = ""
    is_bot_blocked = False

    # Strategy 1: Trafilatura
    if HAS_TRAFILATURA:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted_text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    no_fallback=False
                )
                metadata = trafilatura.extract_metadata(downloaded)
                
                if metadata and metadata.title:
                    title = metadata.title
                if extracted_text:
                    text = extracted_text
        except Exception:
            pass

    # Strategy 2: Requests session + BeautifulSoup
    if not text or len(text) < 50:
        try:
            session = requests.Session()
            response = session.get(url, headers=HEADERS, timeout=timeout)
            
            if response.status_code in [401, 403]:
                is_bot_blocked = True
                domain_name = re.findall(r'https?://(?:www\.)?([^/]+)', url)
                domain_str = domain_name[0] if domain_name else "This site"
                return {
                    "success": False,
                    "title": "",
                    "text": "",
                    "url": url,
                    "is_bot_blocked": True,
                    "error": (
                        f"HTTP {response.status_code} Forbidden/Unauthorized: {domain_str} uses automated "
                        f"anti-bot security (Akamai/Cloudflare/PerimeterX) or a subscription paywall that blocks scrapers. "
                        f"Please copy the article text and paste it into the 'Paste Headline & Text' tab to analyze it."
                    )
                }

            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            if not title:
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                elif soup.find("h1"):
                    title = soup.find("h1").get_text(strip=True)

            # Clean markup
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                element.decompose()

            article_tag = soup.find("article") or soup.find("main")
            if article_tag:
                paragraphs = article_tag.find_all("p")
            else:
                paragraphs = soup.find_all("p")

            extracted_paras = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 25]
            text = "\n\n".join(extracted_paras)

        except requests.exceptions.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            is_blocked = status in [401, 403]
            return {
                "success": False,
                "title": title,
                "text": "",
                "url": url,
                "is_bot_blocked": is_blocked,
                "error": (
                    f"HTTP {status} Blocked: The publisher's server blocked automated scraping. "
                    f"Please copy and paste the article text directly."
                    if is_blocked else str(he)
                )
            }
        except Exception as e:
            if not text:
                return {
                    "success": False,
                    "title": title,
                    "text": "",
                    "url": url,
                    "is_bot_blocked": False,
                    "error": f"Failed to fetch or parse URL: {str(e)}"
                }

    if not text or len(text.strip()) < 30:
        return {
            "success": False,
            "title": title,
            "text": "",
            "url": url,
            "is_bot_blocked": False,
            "error": "Could not extract article text from the URL. The page may require JavaScript or login."
        }

    return {
        "success": True,
        "title": title.strip(),
        "text": text.strip(),
        "url": url,
        "is_bot_blocked": False,
        "error": None
    }
