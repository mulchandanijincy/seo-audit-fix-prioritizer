import datetime, ipaddress, json, re, socket, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from urllib.parse import urljoin, urlparse
import requests
from google import genai
from google.genai import types

_client = genai.Client()  # reads GEMINI_API_KEY automatically


def ask(system, user, tokens=2000):
    r = _client.models.generate_content(
        model="gemini-2.5-flash", contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, response_mime_type="application/json",
            max_output_tokens=tokens, thinking_config=types.ThinkingConfig(thinking_budget=0)))
    return json.loads(r.text)


MAX_PAGES = 8
UA = {"user-agent": "SEOAuditBot/1.0 (student project)"}
SYSTEM = """You are a senior technical SEO. You receive issues found by an automated crawler, each with evidence.
Rules:
- Use ONLY the supplied issues and evidence. Never invent pages, numbers or facts about the business.
- Rank by impact on search visibility first, then by ease of fixing.
- For title or meta description issues, write a corrected example using only text visible in the evidence; if the evidence is not enough, say what to write instead of inventing it.
- Merge repeated issues across pages into one entry listing the affected pages.
Return ONLY JSON: {"summary":"","priority":[{"rank":1,"rule":"","pages":[""],"why":"","effort":"Low|Medium|High","correction":""}]}"""


def is_public(url):
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        return all(ipaddress.ip_address(i[4][0]).is_global for i in socket.getaddrinfo(u.hostname, None))
    except Exception:
        return False


def get(url):
    t = time.time()
    r = requests.get(url, headers=UA, timeout=6)
    if not is_public(r.url):
        raise ValueError("Redirected to a non-public address")
    html = r.text if "text/html" in r.headers.get("content-type", "") else ""
    return {"url": url, "final": r.url, "status": r.status_code, "ms": int((time.time() - t) * 1000), "html": html}


def safe_get(u):
    try:
        return get(u)
    except Exception:
        return {"url": u, "final": u, "status": 0, "ms": 0, "html": ""}


def meta(h, name):
    for m in re.findall(r"<meta\s[^>]*>", h, re.I):
        if re.search(rf'(name|property)=["\']{name}["\']', m, re.I):
            c = re.search(r'content=["\']([^"\']*)', m, re.I)
            return c.group(1) if c else ""
    return None


def parse(p):
    h = p["html"]
    text = re.sub(r"<[^>]+>", " ", re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", h, flags=re.I))
    imgs = re.findall(r"<img\b[^>]*>", h, re.I)
    t = re.search(r"<title[^>]*>([\s\S]*?)</title>", h, re.I)
    can = re.search(r'<link[^>]+rel=["\']canonical["\'][^>]*href=["\']([^"\']+)', h, re.I)
    return {"title": t.group(1).strip() if t else "", "desc": meta(h, "description"), "robots": meta(h, "robots") or "",
            "viewport": meta(h, "viewport"), "ogTitle": meta(h, "og:title"), "canonical": can.group(1) if can else "",
            "lang": bool(re.search(r"<html[^>]*\slang=", h, re.I)), "h1": len(re.findall(r"<h1[\s>]", h, re.I)),
            "noAlt": sum(1 for i in imgs if not re.search(r"\balt=", i, re.I)), "imgs": len(imgs), "words": len(text.split())}


def links(p, host):
    out = []
    for m in re.finditer(r'<a\s[^>]*href=["\']([^"\'#]+)["\']', p["html"], re.I):
        u = urlparse(urljoin(p["final"], m.group(1)))
        if u.hostname == host and u.scheme in ("http", "https") and not re.search(r"\.(pdf|jpg|png|zip|svg|webp)$", u.path, re.I):
            href = u._replace(fragment="").geturl()
            if href not in out:
                out.append(href)
    return out


def disallowed(txt):
    rules, on = [], False
    for line in txt.splitlines():
        l = line.split("#")[0].strip()
        if re.match(r"user-agent:\s*\*", l, re.I):
            on = True
        elif re.match(r"user-agent:", l, re.I):
            on = False
        elif on and l.lower().startswith("disallow:"):
            v = l.split(":", 1)[1].strip()
            if v:
                rules.append(v)
    return rules


def check(p, s, issues):
    def add(sev, rule, ev, fix):
        issues.append({"sev": sev, "rule": rule, "page": p["url"], "evidence": ev, "fix": fix})
    if p["status"] >= 400: add("Critical", "Page returns an error", f'HTTP {p["status"]}', "Fix the page or redirect it to a relevant live page.")
    if "noindex" in s["robots"].lower(): add("Critical", "Page blocked from indexing", f'meta robots = "{s["robots"]}"', "Remove noindex if this page should appear in search.")
    if not s["title"]: add("High", "Missing title tag", "No <title> found", "Add a unique title of 30–60 characters.")
    elif len(s["title"]) > 60: add("Medium", "Title too long", f'{len(s["title"])} characters: "{s["title"]}"', "Shorten to 60 characters or fewer.")
    elif len(s["title"]) < 20: add("Medium", "Title too short", f'{len(s["title"])} characters: "{s["title"]}"', "Describe the page in 30–60 characters.")
    if not s["desc"]: add("High", "Missing meta description", "No meta description found", "Add a 70–155 character summary of the page.")
    elif len(s["desc"]) > 160: add("Low", "Meta description too long", f'{len(s["desc"])} characters', "Shorten to about 155 characters.")
    if s["h1"] == 0: add("High", "Missing H1", "No <h1> found", "Add one H1 that states the page topic.")
    if s["h1"] > 1: add("Medium", "Multiple H1 tags", f'{s["h1"]} <h1> tags', "Keep one H1 and use H2–H3 for sections.")
    if s["noAlt"]: add("Medium", "Images without alt text", f'{s["noAlt"]} of {s["imgs"]} images have no alt attribute', "Add short descriptive alt text to each meaningful image.")
    if not s["canonical"]: add("Medium", "Missing canonical tag", "No rel=canonical link", "Add a canonical link pointing to the preferred URL.")
    if s["viewport"] is None: add("High", "Missing viewport tag", "No meta viewport", 'Add <meta name="viewport" content="width=device-width, initial-scale=1">.')
    if not s["lang"]: add("Low", "Missing html lang", "<html> has no lang attribute", 'Add lang="en" (or the page language).')
    if s["ogTitle"] is None: add("Low", "Missing Open Graph tags", "No og:title", "Add og:title, og:description and og:image for better sharing.")
    if s["words"] < 200 and p["status"] < 400: add("Medium", "Thin content", f'{s["words"]} words of visible text', "Add useful content that answers the visitor’s questions.")
    if p["ms"] > 2000: add("Medium", "Slow server response", f'{p["ms"]} ms to receive the page', "Check hosting, caching and page weight.")
    if p["final"].startswith("http://"): add("High", "Page served over HTTP", p["final"], "Serve over HTTPS and redirect HTTP to HTTPS.")


class handler(BaseHTTPRequestHandler):
    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        try:
            d = json.loads(self.rfile.read(int(self.headers.get("content-length", 0) or 0)) or b"{}")
        except ValueError:
            return self.send(400, {"error": "Invalid JSON"})
        url = d.get("url", "")
        if not d.get("consent"):
            return self.send(400, {"error": "Confirm you have permission to audit this site."})
        if not is_public(url):
            return self.send(400, {"error": "Enter a valid public http(s) URL."})
        try:
            start = urlparse(url)
            origin = f"{start.scheme}://{start.netloc}"
            blocked = []
            try:
                blocked = disallowed(requests.get(origin + "/robots.txt", headers=UA, timeout=4).text)
            except Exception:
                pass
            ok = lambda u: not any(urlparse(u).path.startswith(x) for x in blocked)
            first = get(url)
            queue = [u for u in links(first, start.hostname) if ok(u)][:MAX_PAGES - 1]
            with ThreadPoolExecutor(8) as ex:
                rest = list(ex.map(safe_get, queue))
            pages = [first] + rest
            issues, titles = [], {}
            for p in pages:
                if not p["status"]:
                    issues.append({"sev": "Critical", "rule": "Page failed to load", "page": p["url"], "evidence": "Request timed out or failed", "fix": "Check the URL and server availability."})
                    continue
                s = parse(p)
                check(p, s, issues)
                if s["title"]:
                    titles.setdefault(s["title"], []).append(p["url"])
            for t, v in titles.items():
                if len(v) > 1:
                    issues.append({"sev": "Medium", "rule": "Duplicate title", "page": ", ".join(v), "evidence": f'"{t}" used on {len(v)} pages', "fix": "Give every page a unique title."})
            order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
            issues.sort(key=lambda i: order[i["sev"]])
            ai = None
            try:
                ai = ask(SYSTEM, json.dumps(issues[:60]), 2500)
            except Exception:
                pass
            self.send(200, {"site": origin, "crawled": len(pages), "skippedByRobots": bool(blocked), "date": datetime.date.today().isoformat(), "issues": issues, "ai": ai})
        except Exception as e:
            self.send(500, {"error": str(e)})