"""Turn FastAPI's MkDocs Markdown into clean text for embedding.

The source is not plain Markdown. It contains:
  - code includes:  {* ../../docs_src/body/tutorial001_py310.py ln[1:11] hl[7] *}
                    {!> ../../docs_src/python_types/tutorial009_py310.py!}
  - heading anchors: ## Create your data model { #create-your-data-model }
  - admonitions and tabs:  /// tip | Title  ...  ///     //// tab | Python 3.10+  ...  ////
  - inline HTML: <abbr title="...">ORM</abbr>, <dfn>, <img>, terminal colour <font> tags

Everything here is pure functions over strings so it can be unit-tested
without the real repo.
"""
import html
import re
from dataclasses import dataclass, field
from pathlib import Path

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
INCLUDE_STAR = re.compile(r"\{\*\s*(\S+)((?:\s+[a-z]+\[[^\]]*\])*)\s*\*\}")
INCLUDE_BANG = re.compile(r"\{!>?\s*(\S+?)\s*!\}")
LINE_RANGE = re.compile(r"ln\[([^\]]*)\]")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([\w-]+)\s*\})?\s*$")
ADMONITION = re.compile(r"^(/{3,4})\s*([\w-]+)?\s*(?:\|\s*(.*))?$")
TITLED_TAG = re.compile(r"<(abbr|dfn)\s+title=\"([^\"]*)\"[^>]*>(.*?)</\1>", re.S)
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
JINJA_RAW = re.compile(r"\{%\s*(?:end)?raw\s*%\}")   # mkdocs-macros escaping, not content
# Embedded binary (e.g. a base64 PNG in an example) is noise to an embedding
# model and wastes context window, so long runs are replaced with a marker.
BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

LANG_BY_EXT = {".py": "python", ".html": "html", ".js": "javascript", ".json": "json",
               ".toml": "toml", ".yml": "yaml", ".yaml": "yaml", ".sh": "bash", ".txt": "text"}
SHELL_LANGS = {"console", "bash", "sh", "shell"}


@dataclass
class IncludeStats:
    resolved: int = 0
    unresolved: list[str] = field(default_factory=list)


# ------------------------------------------------------------ code includes

def select_lines(code: str, spec: str) -> str:
    """Apply an ln[...] spec like '1:11' or '1:3,14:18' (1-based, inclusive)."""
    lines = code.splitlines()
    parts = []
    for rng in spec.split(","):
        rng = rng.strip()
        if not rng:
            continue
        start, _, end = rng.partition(":")
        lo = int(start)
        hi = int(end) if end else lo
        parts.append("\n".join(lines[lo - 1:hi]))
    return "\n# ...\n".join(parts)


def resolve_includes(text: str, base_dir: Path, stats: IncludeStats) -> str:
    """Inline the code that the docs pull in from docs_src/.

    Dropping these markers would silently remove every code example, which is
    the most useful content in a developer-docs corpus.
    """
    def read(rel_path: str) -> str | None:
        path = (base_dir / rel_path).resolve()
        if not path.is_file():
            stats.unresolved.append(rel_path)
            return None
        stats.resolved += 1
        return path.read_text(encoding="utf-8").rstrip("\n")

    def star(match):
        rel_path, opts = match.group(1), match.group(2) or ""
        code = read(rel_path)
        if code is None:
            return f"[missing code example: {rel_path}]"
        line_spec = LINE_RANGE.search(opts)
        if line_spec:
            code = select_lines(code, line_spec.group(1))
        lang = LANG_BY_EXT.get(Path(rel_path).suffix, "")
        return f"```{lang}\n{code}\n```"

    def bang(match):  # already sits inside a fence in the source
        code = read(match.group(1))
        return code if code is not None else f"[missing code example: {match.group(1)}]"

    return INCLUDE_BANG.sub(bang, INCLUDE_STAR.sub(star, text))


# ------------------------------------------------------------ prose cleanup

def clean_prose(text: str) -> str:
    text = HTML_COMMENT.sub("", text)
    text = JINJA_RAW.sub("", text)
    # Keep the expansion: "ORM (Object-Relational Mapper)" helps retrieval.
    text = TITLED_TAG.sub(lambda m: f"{m.group(3)} ({m.group(2)})", text)
    text = MD_IMAGE.sub("", text)
    text = HTML_TAG.sub("", text)
    return html.unescape(text)


def convert_admonition(line: str) -> str | None:
    """'/// tip | Title' -> 'Tip: Title'. Closing '///' -> ''. Not one -> None."""
    match = ADMONITION.match(line.strip())
    if not match:
        return None
    _, kind, title = match.groups()
    if not kind:
        return ""                                   # closing marker
    if kind == "tab":
        return f"{title.strip()}:" if title else ""
    label = kind.capitalize()
    return f"{label}: {title.strip()}" if title else f"{label}:"


def slugify(title: str) -> str:
    return re.sub(r"[^\w\s-]", "", title.lower()).strip().replace(" ", "-")


# ------------------------------------------------------------ main entry

def clean_markdown(raw: str, base_dir: Path, stats: IncludeStats) -> tuple[str, list[dict]]:
    """Return (clean text, headings). Each heading records its character offset
    in the clean text, so Step 2 can chunk by section and cite section URLs."""
    text = resolve_includes(raw, base_dir, stats)

    out_lines, headings = [], []
    offset = 0
    fence, fence_lang = None, ""

    for line in text.splitlines():
        fence_match = FENCE.match(line)
        if fence:
            if fence_match and fence_match.group(1)[0] == fence[0] and len(fence_match.group(1)) >= len(fence):
                fence = None
            elif fence_lang in SHELL_LANGS:
                line = html.unescape(HTML_TAG.sub("", line))   # strip terminal colour tags
            # Code is otherwise kept byte-for-byte, except embedded binary blobs.
            line = BASE64_BLOB.sub(lambda m: f"<base64 data, {len(m.group())} chars omitted>", line)
        elif fence_match:
            fence = fence_match.group(1)
            fence_lang = line.strip()[len(fence):].split(" ")[0].lower()
        else:
            admonition = convert_admonition(line)
            if admonition is not None:
                line = admonition
            else:
                line = clean_prose(line)
                heading = HEADING.match(line)
                if heading:
                    hashes, title, anchor = heading.groups()
                    line = f"{hashes} {title}"
                    headings.append({"level": len(hashes), "title": title,
                                     "anchor": anchor or slugify(title), "offset": offset})
            # Outside code, keep at most one blank line in a row (and none at
            # the start). Doing it here keeps heading offsets exact.
            if not line.strip() and (not out_lines or not out_lines[-1].strip()):
                continue
        out_lines.append(line)
        offset += len(line) + 1

    return "\n".join(out_lines).rstrip(), headings
