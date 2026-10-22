"""Loading support tickets from CSV, JSON or JSON Lines exports.

Support tools export in different shapes, so columns are found by name:
the first column whose name matches one of the candidates below is used.
Anything can be overridden with explicit column names.
"""
from __future__ import annotations

import csv
import json
import re
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


def parse_time(value: str | None) -> datetime | None:
    """ISO 8601 dates, 'YYYY-MM-DD HH:MM:SS', or Unix seconds/milliseconds. Naive times are taken as UTC."""
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
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Common export shapes: {"tickets": [...]}, {"conversations": [...]}, {"data": [...]}
            for key in ("tickets", "conversations", "data", "items", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
            raise IngestError("JSON file must be a list of tickets, or an object with a 'tickets' list")
        if isinstance(data, list):
            return data
    raise IngestError(f"unsupported file type {suffix!r}: use .csv, .json or .jsonl")


def load_tickets(
    path: str | Path,
    text_col: str | None = None,
    time_col: str | None = None,
    customer_col: str | None = None,
    label_col: str | None = None,
    limit: int | None = None,
) -> list[Ticket]:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"file not found: {path}")
    rows = _rows(path)
    if not rows:
        raise IngestError(f"{path} has no rows")
    columns = list(rows[0].keys())
    tcol = _pick(columns, TEXT_COLUMNS, text_col)
    if tcol is None:
        raise IngestError(f'no text column found (looked for {", ".join(TEXT_COLUMNS)}); name it with --text-col')
    timecol = _pick(columns, TIME_COLUMNS, time_col)
    idcol = _pick(columns, ID_COLUMNS, None)
    custcol = _pick(columns, CUSTOMER_COLUMNS, customer_col)
    labcol = _pick(columns, LABEL_COLUMNS, label_col)

    out: list[Ticket] = []
    for i, row in enumerate(rows):
        text = str(row.get(tcol) or "").strip()
        if not text:
            continue
        out.append(
            Ticket(
                id=str(row.get(idcol) or i + 1) if idcol else str(i + 1),
                text=text,
                created=parse_time(row.get(timecol)) if timecol else None,
                customer=(str(row.get(custcol)).strip() or None) if custcol and row.get(custcol) is not None else None,
                label=(str(row.get(labcol)).strip() or None) if labcol and row.get(labcol) is not None else None,
            )
        )
        if limit and len(out) >= limit:
            break
    if not out:
        raise IngestError(f"no non-empty text in column {tcol!r}")
    return out


_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_LONGNUM = re.compile(r"\b\d{5,}\b")


def redact(text: str) -> str:
    """Removes personal data before text is shown, stored in a report, or sent to an AI model."""
    text = _PLACEHOLDER.sub(lambda m: "[" + m.group(0).strip("{} ").lower() + "]", text)
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[link]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONGNUM.sub("[number]", text)


def clean(text: str) -> str:
    """Normalises text for analysis: template placeholders, personal data and numbers are removed."""
    text = _PLACEHOLDER.sub(" ", text)
    text = redact(text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()
