"""Self-contained Markdown -> HTML docs renderer (zero dependencies).

One renderer, two consumers:
  * the ``/docs`` route in the dashboard (live-renders ``Guide.md``)
  * ``python -m agentlens.server.docs`` (writes a standalone ``docs/index.html``)

It implements just the slice of GitHub-Flavored Markdown that ``Guide.md`` uses
(ATX headings, pipe tables, fenced code, blockquotes, lists, inline code/bold/
italic/links, horizontal rules). Heading slugs match GitHub's algorithm so the
in-document table-of-contents links resolve.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# slug (GitHub-compatible): lower, drop punctuation except space/_/-, space->-  #
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)   # keep [a-z0-9_], spaces and hyphens
    s = s.replace(" ", "-")
    return s


# --------------------------------------------------------------------------- #
# inline                                                                        #
# --------------------------------------------------------------------------- #
def _plain(s: str) -> str:
    """Strip inline markdown to plain text (for sidebar/TOC labels)."""
    s = re.sub(r"`([^`]+)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s


def _inline(text: str) -> str:
    """Render inline markdown on a single line of *raw* text -> safe HTML."""
    # 1. pull out code spans first so their contents are never further parsed
    spans: List[str] = []

    def _stash(m: "re.Match[str]") -> str:
        spans.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)

    # 2. escape everything else
    text = html.escape(text)

    # 3. links [text](url)
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>',
        text,
    )
    # 4. bold then italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\w)", r"<em>\1</em>", text)

    # 5. restore code spans
    text = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)
    return text


# --------------------------------------------------------------------------- #
# block parser                                                                  #
# --------------------------------------------------------------------------- #
_FENCE = re.compile(r"^```(\w*)\s*$")
_HR = re.compile(r"^(?:---|\*\*\*|___)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_OL = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_UL = re.compile(r"^(\s*)[-*]\s+(.*)$")


def render_markdown(md: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Return ``(html_body, toc)`` where toc is a list of ``(level, slug, text)``
    for every ``##``/``###`` heading (sidebar source)."""
    lines = md.replace("\r\n", "\n").split("\n")
    out: List[str] = []
    toc: List[Tuple[int, str, str]] = []
    seen_slugs: dict = {}
    i, n = 0, len(lines)

    def uniq(slug: str) -> str:
        if slug not in seen_slugs:
            seen_slugs[slug] = 0
            return slug
        seen_slugs[slug] += 1
        return f"{slug}-{seen_slugs[slug]}"

    while i < n:
        line = lines[i]

        # blank
        if not line.strip():
            i += 1
            continue

        # fenced code
        mf = _FENCE.match(line)
        if mf:
            lang = mf.group(1)
            i += 1
            buf: List[str] = []
            while i < n and not _FENCE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            i += 1  # closing fence
            code = html.escape("\n".join(buf))
            label = f'<span class="lang">{html.escape(lang)}</span>' if lang else ""
            out.append(
                f'<div class="codeblock">{label}'
                f'<button class="copy" type="button" aria-label="Copy">Copy</button>'
                f'<pre><code>{code}</code></pre></div>'
            )
            continue

        # heading
        mh = _HEADING.match(line)
        if mh:
            level = len(mh.group(1))
            text = mh.group(2).strip()
            slug = uniq(slugify(text))
            inner = _inline(text)
            out.append(
                f'<h{level} id="{slug}">{inner}'
                f'<a class="anchor" href="#{slug}" aria-label="Link">#</a></h{level}>'
            )
            if level in (2, 3):
                toc.append((level, slug, _plain(text)))
            i += 1
            continue

        # horizontal rule
        if _HR.match(line):
            out.append("<hr>")
            i += 1
            continue

        # table (header row + |---| separator)
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            out.append(_table(lines, i, n))
            # advance past the whole table
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip():
                i += 1
            continue

        # blockquote
        if line.lstrip().startswith(">"):
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(buf))}</blockquote>")
            continue

        # lists
        if _UL.match(line) or _OL.match(line):
            block, i = _list(lines, i, n)
            out.append(block)
            continue

        # paragraph (gather until blank / block start)
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not _is_block_start(lines[i], lines, i, n):
            buf.append(lines[i])
            i += 1
        out.append(f"<p>{_inline(' '.join(b.strip() for b in buf))}</p>")

    return "\n".join(out), toc


def _is_block_start(line: str, lines: List[str], i: int, n: int) -> bool:
    if _HEADING.match(line) or _FENCE.match(line) or _HR.match(line):
        return True
    if line.lstrip().startswith(">") or _UL.match(line) or _OL.match(line):
        return True
    if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
        return True
    return False


def _cells(row: str) -> List[str]:
    row = row.strip().strip("|")
    return [c.strip() for c in row.split("|")]


def _table(lines: List[str], i: int, n: int) -> str:
    head = _cells(lines[i])
    body_rows = []
    j = i + 2
    while j < n and "|" in lines[j] and lines[j].strip():
        body_rows.append(_cells(lines[j]))
        j += 1
    th = "".join(f"<th>{_inline(c)}</th>" for c in head)
    trs = []
    for r in body_rows:
        tds = "".join(f"<td>{_inline(c)}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return (
        '<div class="tablewrap"><table>'
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>"
    )


def _list(lines: List[str], i: int, n: int) -> Tuple[str, int]:
    ordered = bool(_OL.match(lines[i]))
    items: List[str] = []
    while i < n:
        m = _OL.match(lines[i]) if ordered else _UL.match(lines[i])
        if not m:
            # allow the other bullet style to terminate the list
            if _OL.match(lines[i]) or _UL.match(lines[i]) or not lines[i].strip():
                break
            break
        items.append(_inline(m.group(2).strip()))
        i += 1
    tag = "ol" if ordered else "ul"
    lis = "".join(f"<li>{it}</li>" for it in items)
    return f"<{tag}>{lis}</{tag}>", i


# --------------------------------------------------------------------------- #
# page shell (inline CSS + JS so the file is fully self-contained / offline)    #
# --------------------------------------------------------------------------- #
def build_page(md: str, *, title: str = "AgentLens — Guide",
               fonts_href: Optional[str] = None, home_href: Optional[str] = None) -> str:
    body, toc = render_markdown(md)
    nav = []
    for level, slug, text in toc:
        cls = "lvl2" if level == 2 else "lvl3"
        nav.append(f'<a class="{cls}" href="#{slug}" data-slug="{slug}">{html.escape(text)}</a>')
    navhtml = "\n".join(nav)
    fontlink = f'<link rel="stylesheet" href="{fonts_href}">' if fonts_href else ""
    home = (f'<a class="home" href="{html.escape(home_href, quote=True)}">&#8592; Dashboard</a>'
            if home_href else "")
    return _TEMPLATE.format(title=html.escape(title), nav=navhtml, body=body,
                            fontlink=fontlink, home=home, css=_CSS, js=_JS)


_CSS = r"""
:root{
  --bg:#0b0f16; --panel:#111722; --panel2:#0e141d; --border:#1e2735; --border-soft:#171f2b;
  --text:#e7edf5; --muted:#9fb0c3; --faint:#6b7a8d; --accent:#5b9bff; --accent-soft:rgba(91,155,255,.10);
  --good:#46d39a; --bad:#ff6b6b; --radius:13px;
  --sans:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",system-ui,sans-serif;
  --mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:15px;line-height:1.65;
  -webkit-font-smoothing:antialiased}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.layout{display:grid;grid-template-columns:288px minmax(0,1fr);max-width:1320px;margin:0 auto;gap:0}
/* sidebar */
aside{position:sticky;top:0;height:100vh;overflow-y:auto;border-right:1px solid var(--border-soft);
  padding:26px 18px 60px;background:linear-gradient(180deg,var(--panel2),var(--bg))}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;margin:0 6px 18px;color:var(--text)}
.brand .logo{color:var(--accent);filter:drop-shadow(0 0 10px rgba(91,155,255,.55));font-size:19px}
.brand small{display:block;font-weight:500;font-size:11.5px;color:var(--faint);letter-spacing:.04em}
.home{display:inline-block;margin:0 6px 14px;font-size:12.5px;color:var(--muted)}
.home:hover{color:var(--accent)}
.search{width:100%;padding:9px 12px;margin:0 0 16px;border-radius:10px;border:1px solid var(--border);
  background:var(--panel);color:var(--text);font-size:13px;font-family:var(--sans);outline:none}
.search:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
nav{display:flex;flex-direction:column;gap:1px}
nav a{display:block;padding:6px 10px;border-radius:8px;color:var(--muted);font-size:13.2px;line-height:1.35;
  border-left:2px solid transparent}
nav a.lvl3{padding-left:24px;font-size:12.6px;color:var(--faint)}
nav a:hover{background:var(--accent-soft);color:var(--text);text-decoration:none}
nav a.active{background:var(--accent-soft);color:var(--text);border-left-color:var(--accent)}
nav a.hidden{display:none}
/* content */
main{padding:42px 56px 120px;max-width:900px}
main h1{font-size:33px;line-height:1.15;letter-spacing:-.02em;margin:0 0 6px}
main h2{font-size:23px;letter-spacing:-.01em;margin:52px 0 14px;padding-top:14px;border-top:1px solid var(--border-soft)}
main h3{font-size:17.5px;margin:30px 0 10px}
main h4{font-size:15px;margin:22px 0 8px;color:var(--muted)}
.anchor{margin-left:10px;color:var(--faint);opacity:0;font-weight:400;text-decoration:none;transition:opacity .12s}
h1:hover .anchor,h2:hover .anchor,h3:hover .anchor{opacity:1}
main p{margin:0 0 14px}
main hr{border:none;border-top:1px solid var(--border-soft);margin:34px 0}
main ul,main ol{margin:0 0 16px;padding-left:24px}
main li{margin:5px 0}
strong{color:#fff;font-weight:650}
code{font-family:var(--mono);font-size:.86em;background:rgba(255,255,255,.055);border:1px solid var(--border-soft);
  border-radius:5px;padding:1.5px 5px;color:#d8c7ff}
blockquote{margin:0 0 16px;padding:12px 18px;border-left:3px solid var(--accent);border-radius:0 10px 10px 0;
  background:var(--accent-soft);color:var(--muted)}
blockquote code{color:#d8c7ff}
/* code blocks */
.codeblock{position:relative;margin:0 0 18px;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--panel2);overflow:hidden}
.codeblock pre{margin:0;padding:16px 18px;overflow-x:auto}
.codeblock code{display:block;background:none;border:none;padding:0;color:#cdd9e7;font-size:13px;line-height:1.6}
.codeblock .lang{position:absolute;top:0;left:0;font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;
  text-transform:uppercase;color:var(--faint);padding:5px 12px;background:var(--panel);border-bottom-right-radius:8px}
.codeblock .lang+button.copy{}
.codeblock:has(.lang) pre{padding-top:30px}
.copy{position:absolute;top:7px;right:8px;font-family:var(--sans);font-size:11px;color:var(--muted);cursor:pointer;
  background:var(--panel);border:1px solid var(--border);border-radius:7px;padding:3px 10px;opacity:0;transition:.12s}
.codeblock:hover .copy{opacity:1}
.copy:hover{color:var(--text);border-color:var(--accent)}
.copy.done{color:var(--good);border-color:var(--good)}
/* tables */
.tablewrap{overflow-x:auto;margin:0 0 18px;border:1px solid var(--border);border-radius:var(--radius)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
thead th{text-align:left;background:var(--panel);color:var(--muted);font-weight:600;font-size:11.5px;
  text-transform:uppercase;letter-spacing:.05em;padding:11px 14px;border-bottom:1px solid var(--border)}
td{padding:10px 14px;border-bottom:1px solid var(--border-soft);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
tbody tr:nth-child(even){background:rgba(255,255,255,.012)}
/* responsive */
.menu-toggle{display:none}
@media(max-width:880px){
  .layout{grid-template-columns:1fr}
  aside{position:fixed;z-index:40;width:280px;left:-300px;transition:left .2s;box-shadow:0 0 40px rgba(0,0,0,.5)}
  aside.open{left:0}
  main{padding:64px 22px 90px}
  .menu-toggle{display:flex;position:fixed;top:14px;left:14px;z-index:50;align-items:center;justify-content:center;
    width:42px;height:38px;border-radius:10px;background:var(--panel);border:1px solid var(--border);color:var(--text);
    cursor:pointer;font-size:18px}
}
"""

_JS = r"""
// copy buttons
document.querySelectorAll('.codeblock .copy').forEach(function(btn){
  btn.addEventListener('click', function(){
    var code = btn.parentElement.querySelector('code').innerText;
    navigator.clipboard.writeText(code).then(function(){
      btn.textContent='Copied'; btn.classList.add('done');
      setTimeout(function(){ btn.textContent='Copy'; btn.classList.remove('done'); }, 1400);
    });
  });
});
// sidebar search filter
var search = document.getElementById('docsearch');
if(search){
  search.addEventListener('input', function(){
    var q = search.value.trim().toLowerCase();
    document.querySelectorAll('nav a').forEach(function(a){
      a.classList.toggle('hidden', q && a.textContent.toLowerCase().indexOf(q)===-1);
    });
  });
}
// scrollspy: highlight nav link of the section in view
var links = [].slice.call(document.querySelectorAll('nav a'));
var map = {}; links.forEach(function(a){ map[a.dataset.slug]=a; });
var heads = links.map(function(a){ return document.getElementById(a.dataset.slug); }).filter(Boolean);
function spy(){
  var top = window.scrollY + 120, cur = null;
  heads.forEach(function(h){ if(h.offsetTop <= top) cur = h.id; });
  links.forEach(function(a){ a.classList.toggle('active', a.dataset.slug===cur); });
}
window.addEventListener('scroll', spy, {passive:true}); spy();
// mobile menu
var aside = document.querySelector('aside'), tog = document.querySelector('.menu-toggle');
if(tog){ tog.addEventListener('click', function(){ aside.classList.toggle('open'); }); }
document.querySelectorAll('nav a').forEach(function(a){
  a.addEventListener('click', function(){ if(aside.classList.contains('open')) aside.classList.remove('open'); });
});
"""

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{fontlink}
<style>{css}</style>
</head>
<body>
<button class="menu-toggle" aria-label="Menu">&#9776;</button>
<div class="layout">
<aside>
  <div class="brand"><span class="logo">&#9678;</span><span>AgentLens<small>The Complete Guide</small></span></div>
  {home}
  <input id="docsearch" class="search" type="search" placeholder="Filter sections…" autocomplete="off">
  <nav>
{nav}
  </nav>
</aside>
<main>
{body}
</main>
</div>
<script>{js}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# locate Guide.md (dev repo root, package data, or cwd)                          #
# --------------------------------------------------------------------------- #
def find_guide() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "Guide.md",                 # bundled into package (if shipped)
        here.parents[2] / "Guide.md",             # repo root (dev)
        Path.cwd() / "Guide.md",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def guide_html(fonts_href: Optional[str] = "/static/fonts.css") -> str:
    g = find_guide()
    if not g:
        return build_page("# Guide not found\n\nCould not locate `Guide.md`.")
    return build_page(g.read_text(encoding="utf-8"), fonts_href=fonts_href, home_href="/")


if __name__ == "__main__":
    # standalone build: docs/index.html (offline, system fonts, no external refs)
    g = find_guide()
    if not g:
        raise SystemExit("Guide.md not found")
    out_dir = g.parent / "docs"
    out_dir.mkdir(exist_ok=True)
    page = build_page(g.read_text(encoding="utf-8"), fonts_href=None)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'} ({len(page):,} bytes)")
