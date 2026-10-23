"""Loading support tickets from CSV, JSON or JSON Lines exports.

Support tools export in different shapes, so columns are found by name:
the first column whose name matches one of the candidates below is used.
Anything can be overridden with explicit column names.
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TEXT_COLUMNS = ["text", "message", "body", "instruction", "description", "content", "ticket", "comment", "question", "subject"]
TIME_COLUMNS = ["created_at", "created", "timestamp", "date", "time", "submitted_at", "opened_at"]
ID_COLUMNS = ["id", "ticket_id", "conversation_id", "case_id"]
CUSTOMER_COLUMNS = ["customer", "customer_id", "account", "account_id", "user_id", "requester", "company"]
LABEL_COLUMNS = ["intent", "label", "category", "topic", "tag"]


@dataclass
class Ticket:
    id: str
    text: str
    created: datetime | None = None
    customer: str | None = None
    #: A known label, if the export has one. Only used for evaluation, never for analysis.
    label: str | None = None
    extra: dict = field(default_factory=dict)


class IngestError(ValueError):
    pass


def _pick(columns: list[str], candidates: list[str], explicit: str | None) -> str | None:
    if explicit:
        if explicit not in columns:
            raise IngestError(f'column "{explicit}" not found; columns are: {", ".join(columns)}')
        return explicit
    lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


_MONTHS = "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%d-%b-%Y"
_SLASH = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?\s*$")


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_time(value: str | None, dayfirst: bool | None = None) -> datetime | None:
    """One timestamp. ISO 8601, Unix seconds/milliseconds, "Sep 1, 2026", or d/m/y and m/d/y
    (for those, `dayfirst` says which; if None, only unambiguous dates parse). Naive times are UTC."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{9,13}(\.\d+)?", s):
        n = float(s)
        if n > 1e11:  # milliseconds
            n /= 1000
        return datetime.fromtimestamp(n, tz=timezone.utc)
    try:
        return _aware(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        pass
    m = _SLASH.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        if dayfirst is None:
            if a > 12 >= b:
                dayfirst = True
            elif b > 12 >= a:
                dayfirst = False
            else:
                return None  # ambiguous on its own; the column decides (see detect_dayfirst)
        day, month = (a, b) if dayfirst else (b, a)
        try:
            return datetime(y, month, day, int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0), tzinfo=timezone.utc)
        except ValueError:
            return None
    for fmt in _MONTHS:
        try:
            return _aware(datetime.strptime(s, fmt))
        except ValueError:
            continue
    return None


def detect_dayfirst(values: list[str]) -> bool | None:
    """Decides day/month order for a whole column, so 04/12 and 04/13 can't be read differently.

    Returns True (d/m/y), False (m/d/y), or None when the column has no slash dates.
    Raises IngestError when the column contains both orders or can't be told apart.
    """
    first_big = second_big = slash = False
    for v in values:
        m = _SLASH.match(str(v or ""))
        if not m:
            continue
        slash = True
        a, b = int(m.group(1)), int(m.group(2))
        first_big |= a > 12
        second_big |= b > 12
    if not slash:
        return None
    if first_big and second_big:
        raise IngestError("the date column mixes day/month and month/day order; fix the export or pass --dayfirst / --monthfirst")
    if first_big:
        return True
    if second_big:
        return False
    raise IngestError("dates like 03/04/2026 are ambiguous (3 April or March 4?); pass --dayfirst or --monthfirst")


def read_text(path: Path, encoding: str | None = None) -> str:
    """Reads a text file the way Windows tools write them: UTF-8 with or without BOM, UTF-16/32 with BOM
    (PowerShell's Out-File), and Excel's Windows-1252 CSVs as a fallback."""
    raw = path.read_bytes()
    if encoding:
        return raw.decode(encoding)
    for bom, enc in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if raw.startswith(bom):
            return raw.decode(enc)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        print(f"Note: {path.name} isn't UTF-8; reading it as Windows-1252 (use --encoding to override).", file=sys.stderr)
        return raw.decode("cp1252", errors="replace")


def _rows(path: Path, encoding: str | None = None) -> list[dict]:
    suffix = path.suffix.lower()
    text = read_text(path, encoding)
    if suffix == ".csv":
        return list(csv.DictReader(io.StringIO(text, newline="")))
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            # Common export shapes: {"tickets": [...]}, {"conversations": [...]}, {"data": [...]}
            for key in ("tickets", "conversations", "data", "items", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                raise IngestError("JSON file must be a list of tickets, or an object with a 'tickets' list")
        if not isinstance(data, list):
            raise IngestError("JSON file must be a list of tickets")
        rows = data
    else:
        raise IngestError(f"unsupported file type {suffix!r}: use .csv, .json or .jsonl")
    out = []
    for r in rows:
        if isinstance(r, str):
            out.append({"text": r})  # a plain list of messages
        elif isinstance(r, dict):
            out.append(r)
        else:
            raise IngestError("every ticket must be an object (or a plain string)")
    return out


@dataclass
class LoadStats:
    rows: int = 0
    skipped_empty: int = 0
    dates_found: int = 0
    dates_failed: int = 0
    columns: dict = field(default_factory=dict)


class TicketList(list):
    """A list of tickets that also carries what happened while loading them."""

    stats: LoadStats


def load_tickets(
    path: str | Path,
    text_col: str | None = None,
    time_col: str | None = None,
    customer_col: str | None = None,
    label_col: str | None = None,
    limit: int | None = None,
    dayfirst: bool | None = None,
    encoding: str | None = None,
) -> TicketList:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    try:
        rows = _rows(path, encoding)
    except (json.JSONDecodeError, UnicodeError, csv.Error) as e:
        raise IngestError(f"couldn't read {path.name}: {e}") from e
    if not rows:
        raise IngestError(f"{path} has no rows")
    # Columns from the first rows together, so a field missing in row 1 is still found.
    columns: list[str] = []
    for r in rows[:500]:
        for c in r.keys():
            if c not in columns:
                columns.append(str(c))
    tcol = _pick(columns, TEXT_COLUMNS, text_col)
    if tcol is None:
        raise IngestError(f'no text column found (looked for {", ".join(TEXT_COLUMNS)}); name it with --text-col')
    timecol = _pick(columns, TIME_COLUMNS, time_col)
    idcol = _pick(columns, ID_COLUMNS, None)
    custcol = _pick(columns, CUSTOMER_COLUMNS, customer_col)
    labcol = _pick(columns, LABEL_COLUMNS, label_col)
    if timecol and dayfirst is None:
        dayfirst = detect_dayfirst([r.get(timecol) for r in rows])

    def val(row: dict, col: str | None) -> str | None:
        if not col:
            return None
        v = row.get(col)
        if v is None:
            return None
        v = str(v).strip()
        return v or None

    out = TicketList()
    stats = LoadStats(rows=len(rows), columns={"text": tcol, "time": timecol, "id": idcol, "customer": custcol, "label": labcol})
    seen_ids: dict[str, int] = {}
    for i, row in enumerate(rows):
        text = val(row, tcol)
        if not text:
            stats.skipped_empty += 1
            continue
        tid = val(row, idcol) or f"row-{i + 1}"
        if tid in seen_ids:  # keep ids unique, so every ticket can be referred to
            seen_ids[tid] += 1
            tid = f"{tid}#{seen_ids[tid]}"
        else:
            seen_ids[tid] = 1
        created = None
        raw_time = val(row, timecol)
        if raw_time:
            created = parse_time(raw_time, dayfirst)
            if created:
                stats.dates_found += 1
            else:
                stats.dates_failed += 1
        out.append(Ticket(id=tid, text=text, created=created, customer=val(row, custcol), label=val(row, labcol)))
        if limit and len(out) >= limit:
            break
    if not out:
        raise IngestError(f"no non-empty text in column {tcol!r}")
    out.stats = stats
    return out


_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
_EMAIL = re.compile(r"[\w.+-]+\s*(?:@|\(at\)|\[at\]|\sat\s)\s*[\w-]+\s*(?:\.|\(dot\)|\[dot\])\s*[\w.-]+", re.I)
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|in|co|app|dev|info|biz|me)/\S*", re.I)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<![\w])\+?\d[\d\s().-]{5,}\d(?![\w])")
_LONGNUM = re.compile(r"[A-Za-z]{0,4}#?(?<!\d)\d{5,}")
_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")  # UK-style


def redact(text: str) -> str:
    """Removes common personal data before text is shown, stored in a report, or sent to an AI model:
    emails (including "name(at)site.com"), links, card and bank numbers, phone numbers, long numbers
    and account references, and UK-style postcodes.

    Pattern-based, so it can't catch everything: names and street addresses written in plain words
    get through. Treat reports as containing customer data.
    """
    text = _PLACEHOLDER.sub(lambda m: "[" + m.group(0).strip("{} ").lower() + "]", text)
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    text = _IBAN.sub("[account]", text)
    text = _CARD.sub("[number]", text)
    text = _PHONE.sub("[phone]", text)
    text = _LONGNUM.sub("[number]", text)
    return _POSTCODE.sub("[postcode]", text)


def clean(text: str) -> str:
    """Normalises text for analysis: template placeholders, personal data and numbers are removed."""
    text = _PLACEHOLDER.sub(" ", text)
    text = redact(text)
    text = re.sub(r"\[(?:email|link|account|number|phone|postcode)\]|\[[a-z ]{1,30}\]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
