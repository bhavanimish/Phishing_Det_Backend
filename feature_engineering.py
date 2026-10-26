import re
from urllib.parse import urlparse

IP_PATTERN = re.compile(
    r'^(?:http[s]?://)?'  # optional scheme
    r'(?:(?:\d{1,3}\.){3}\d{1,3})'  # IPv4
    r'(?::\d+)?'  # optional port
    r'(?:[/?#]|$)'
)

def extract_features(url: str) -> dict:
    """
    Extracts 18 numerical features from a raw URL.
    """
    url_str = str(url) if url is not None else ""
    url_lower = url_str.lower()

    # 1. URL structure features (9)
    url_length = len(url_str)
    num_dots = url_str.count('.')
    num_hyphens = url_str.count('-')
    num_slashes = url_str.count('/')
    num_at = url_str.count('@')
    num_percent = url_str.count('%')
    num_question = url_str.count('?')
    num_equals = url_str.count('=')
    num_ampersand = url_str.count('&')

    # 2. Security/domain features (4)
    uses_https = 1 if url_lower.startswith('https://') else 0
    
    # Check IP
    has_ip = 1 if IP_PATTERN.match(url_lower) else 0

    # Parse domain
    parsed = urlparse(url_str if url_lower.startswith(('http://', 'https://')) else 'http://' + url_str)
    netloc = parsed.netloc or parsed.path.split('/')[0]
    hostname = netloc.split(':')[0]  # remove port if present
    
    domain_length = len(hostname)

    # Subdomain count
    if has_ip:
        num_subdomains = 0
    else:
        parts = [p for p in hostname.split('.') if p]
        # Basic heuristic: if parts >= 2, subdomains = len(parts) - 2 (e.g., sub.domain.com -> 3 - 2 = 1)
        num_subdomains = max(0, len(parts) - 2)

    # 3. Suspicious keywords features (5, case-insensitive)
    has_login = 1 if 'login' in url_lower else 0
    has_verify = 1 if 'verify' in url_lower else 0
    has_secure = 1 if 'secure' in url_lower else 0
    has_account = 1 if 'account' in url_lower else 0
    has_update = 1 if 'update' in url_lower else 0

    return {
        'url_length': url_length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_slashes': num_slashes,
        'num_at': num_at,
        'num_percent': num_percent,
        'num_question': num_question,
        'num_equals': num_equals,
        'num_ampersand': num_ampersand,
        'uses_https': uses_https,
        'has_ip': has_ip,
        'num_subdomains': num_subdomains,
        'domain_length': domain_length,
        'has_login': has_login,
        'has_verify': has_verify,
        'has_secure': has_secure,
        'has_account': has_account,
        'has_update': has_update
    }
