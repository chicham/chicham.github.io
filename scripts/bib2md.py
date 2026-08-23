#!/usr/bin/env python3
"""Render publications.bib into the markdown included by publications.qmd.

The .bib file is the single source of truth for the publication list. Entries
keep the citation keys used by the LaTeX CV, so the same file can feed both.

Recognised beyond standard BibTeX: `code` (repository URL). Unknown fields are
ignored by biber, so adding them here does not affect the CV build.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "publications.bib"
OUT = ROOT / "_publications-generated.md"

ME = "Hicham Randrianarivo"

# LaTeX escapes that appear in bibliographic names and titles.
ACCENTS = {
    r"\'e": "é", r"\'E": "É", r"\'a": "á", r"\'o": "ó", r"\'u": "ú", r"\'i": "í",
    r"\`e": "è", r"\`a": "à", r"\`u": "ù",
    r'\"e': "ë", r'\"i': "ï", r'\"o': "ö", r'\"u': "ü", r'\"a': "ä",
    r"\^e": "ê", r"\^a": "â", r"\^i": "î", r"\^o": "ô", r"\^u": "û",
    r"\c c": "ç", r"\~n": "ñ", r"\ss": "ß",
}

MONTHS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}


def delatex(text: str) -> str:
    """Turn a BibTeX field value into display text."""
    for pattern, char in ACCENTS.items():
        text = text.replace("{" + pattern + "}", char).replace(pattern, char)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("--", "–")
    return re.sub(r"\s+", " ", text).strip()


def split_top_level(text: str, sep: str) -> list[str]:
    """Split on `sep`, ignoring occurrences nested inside braces."""
    parts, depth, current = [], 0, []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0 and text.startswith(sep, i):
            parts.append("".join(current))
            current = []
            i += len(sep)
            continue
        current.append(char)
        i += 1
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def parse_entries(source: str) -> list[dict]:
    """Parse the subset of BibTeX this file uses, keeping each entry's raw text."""
    entries = []
    for match in re.finditer(r"@(\w+)\s*\{", source):
        depth, i = 1, match.end()
        while depth and i < len(source):
            depth += {"{": 1, "}": -1}.get(source[i], 0)
            i += 1
        raw = source[match.start():i]
        body = source[match.end():i - 1]
        key, _, rest = body.partition(",")

        fields = {}
        for chunk in split_top_level(rest, ","):
            name, sep, value = chunk.partition("=")
            if not sep:
                continue
            value = value.strip()
            if value and value[0] in "{\"":
                value = value[1:-1]
            fields[name.strip().lower()] = value

        entries.append(
            {"type": match.group(1).lower(), "key": key.strip(),
             "raw": raw, "fields": fields}
        )
    return entries


def format_author(name: str) -> str:
    """Normalise 'Last, First' and 'First Last' to display order, bolding me."""
    name = delatex(name)
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}".strip()
    return f"**{name}**" if name == ME else name


def format_authors(raw: str) -> str:
    names = [format_author(n) for n in split_top_level(raw, " and ")]
    if names and names[-1] == "others":
        return ", ".join(names[:-1]) + ", et al"
    return ", ".join(names)


def format_venue(entry: dict) -> str:
    """One line naming where the work appeared."""
    f = entry["fields"]
    year = f.get("year", "")

    if entry["type"] == "phdthesis":
        return f"PhD thesis, {delatex(f.get('school', ''))}, {year}."

    venue = delatex(f.get("booktitle") or f.get("journal") or "")
    if not venue and f.get("archiveprefix", "").lower() == "arxiv":
        venue = "arXiv preprint"
    if not venue and f.get("doi"):
        month = MONTHS.get(f.get("month", "").lower(), "")
        return f"Dataset, {month} {year}.".replace("  ", " ")

    bits = [venue]
    if f.get("publisher"):
        bits.append(delatex(f["publisher"]))
    elif f.get("organization"):
        bits.append(delatex(f["organization"]))
    if f.get("volume"):
        vol = f["volume"] + (f"({f['number']})" if f.get("number") else "")
        bits.append(vol)
    bits.append(year)
    if f.get("pages"):
        pages = delatex(f["pages"])
        bits.append(("pp. " if "–" in pages else "p. ") + pages)
    return ", ".join(b for b in bits if b) + "."


def format_links(entry: dict) -> list[str]:
    f = entry["fields"]
    links = []
    if f.get("eprint"):
        links.append(f"[arXiv](https://arxiv.org/abs/{f['eprint']})")
    elif f.get("url", "").startswith("https://arxiv.org"):
        links.append(f"[arXiv]({f['url']})")
    if f.get("doi"):
        links.append(f"[DOI](https://doi.org/{f['doi']})")
    if f.get("code"):
        links.append(f"[code]({f['code']})")
    return [f"{link}{{.pub-link}}" for link in links]


def year_of(entry: dict) -> str:
    """Year the work appeared, which is not the preprint year for accepted papers."""
    f = entry["fields"]
    return f.get("sortyear") or f.get("year", "n.d.")


def sort_key(entry: dict) -> tuple[int, str]:
    return (-int(year_of(entry)),
            delatex(entry["fields"].get("title", "")).lower())


def render(entry: dict) -> str:
    f = entry["fields"]
    note = delatex(f.get("note", ""))
    venue = note if note and note != "PhD thesis" else format_venue(entry).rstrip(".")

    parts = [
        "::: {.pub}",
        "[" + delatex(f.get("title", "")).rstrip(".") + "]{.pub-title}",
        "",
        "[" + format_authors(f.get("author", "")) + "]{.pub-authors}",
        "",
        " ".join(["[" + venue + "]{.pub-venue}"] + format_links(entry)),
        "",
        "<details><summary>BibTeX</summary>",
        "",
        "``` bibtex",
        entry["raw"].strip(),
        "```",
        "",
        "</details>",
        ":::",
    ]
    return "\n".join(parts)


def main() -> None:
    entries = sorted(parse_entries(BIB.read_text(encoding="utf-8")), key=sort_key)

    by_year = defaultdict(list)
    for entry in entries:
        by_year[year_of(entry)].append(entry)

    out = ["<!-- Generated by scripts/bib2md.py from publications.bib. Do not edit. -->"]
    for year in sorted(by_year, reverse=True):
        out += ["", f"## {year}", ""]
        out.append("\n\n".join(render(e) for e in by_year[year]))

    text = "\n".join(out).rstrip() + "\n"
    assert "—" not in text, "em dash in generated output"
    prose = re.sub(r"``` bibtex.*?```", "", text, flags=re.S)
    assert not re.search(r"\\[`'\"^~]", prose), "unrendered LaTeX escape"
    OUT.write_text(text, encoding="utf-8")
    print(f"{len(entries)} entries -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
