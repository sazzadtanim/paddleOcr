import os
import io
import re
import numpy as np
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI()

# Forced English — not configurable via env var
LANG = "en"
MODEL_DIR = os.getenv("MODEL_DIR", "/models")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang=LANG,
    det_model_dir=f"{MODEL_DIR}/det",
    rec_model_dir=f"{MODEL_DIR}/rec",
    cls_model_dir=f"{MODEL_DIR}/cls",
    show_log=False,
)


@app.get("/health")
def health():
    return {"status": "ok", "lang": LANG}


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def run_raw_ocr(img_array):
    """Run PaddleOCR and return list of recognized text lines (top to bottom)."""
    result = ocr.ocr(img_array, cls=True)
    lines = []
    for line in result[0] if result and result[0] else []:
        box, (text, confidence) = line
        y = min(point[1] for point in box)
        lines.append({"text": text, "confidence": float(confidence), "y": y})
    lines.sort(key=lambda l: l["y"])
    return lines


# ---------------------------------------------------------------------------
# MRZ detection
#
# IMPORTANT: PaddleOCR frequently drops the '<' filler glyph entirely (not just
# trailing padding — it can vanish from the middle of a line too), so detection
# and parsing must not assume '<' survived. We detect MRZ lines structurally:
#   line1: "P" + 3-letter country code + letters only (name data), no digits
#   line2: mostly digits/letters with an isolated M/F sex marker, length 30-44
# ---------------------------------------------------------------------------

MRZ_LINE1_RE = re.compile(r"^P[A-Z]{3}[A-Z<]{10,45}$")
MRZ_LINE2_RE = re.compile(r"^[A-Z0-9<]{30,44}$")

# Common OCR misreads when the expected character is a digit
DIGIT_CONFUSION = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}


def fix_numeric_field(s):
    """Correct common letter/digit OCR confusions in a field expected to be all digits (or '<')."""
    return "".join(DIGIT_CONFUSION.get(c, c) if c not in "0123456789<" else c for c in s)


def find_mrz_lines(lines):
    """Return the two MRZ lines (TD3 passport format) if found, else None.
    Lines are NOT force-padded to 44 chars here — callers must handle
    variable length since dropped '<' can shrink either line unpredictably."""
    candidates = [l["text"].upper().replace(" ", "") for l in lines]
    for i in range(len(candidates) - 1):
        l1, l2 = candidates[i], candidates[i + 1]
        if MRZ_LINE1_RE.match(l1) and MRZ_LINE2_RE.match(l2) and re.search(r"\d", l2) and ("M" in l2 or "F" in l2):
            return l1, l2
    return None


# ---------------------------------------------------------------------------
# ICAO 9303 check digit algorithm
# ---------------------------------------------------------------------------

_WEIGHTS = [7, 3, 1]


def _char_value(c):
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    if c.isalpha():
        return ord(c) - 55  # A=10 ... Z=35
    return 0


def check_digit(s):
    total = 0
    for i, c in enumerate(s):
        total += _char_value(c) * _WEIGHTS[i % 3]
    return total % 10


def parse_mrz_date(yymmdd, field_type="birth"):
    """Convert MRZ YYMMDD to YYYY-MM-DD.

    field_type="birth": assumes 00-30 -> 2000s, else 1900s (birthdate heuristic).
    field_type="expiry": always resolves to 20xx, since expiry dates on any
    passport still in circulation fall in the current century.
    """
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = yymmdd[0:2], yymmdd[2:4], yymmdd[4:6]
    if field_type == "expiry":
        century = "20"
    else:
        century = "20" if int(yy) <= 30 else "19"
    return f"{century}{yy}-{mm}-{dd}"


NATIONALITY_MAP = {
    "BGD": "BANGLADESHI", "IND": "INDIAN", "PAK": "PAKISTANI", "USA": "AMERICAN",
    "GBR": "BRITISH", "CAN": "CANADIAN", "AUS": "AUSTRALIAN", "SAU": "SAUDI ARABIAN",
    "ARE": "EMIRATI", "MYS": "MALAYSIAN", "SGP": "SINGAPOREAN",
}


def split_name_blob(blob, raw_lines):
    """Split a run-together MRZ name blob (e.g. 'UDDINMUHAMMEDSHARIF') into
    surname/given_name by cross-referencing standalone alphabetic printed
    lines (e.g. 'UDDIN', 'MUHAMMED SHARIF') that the VIZ text usually has
    correctly separated, even when the MRZ lost its '<<' delimiters."""
    # If '<<' survived, use it directly — most reliable path.
    if "<<" in blob:
        parts = blob.split("<<", 1)
        surname = parts[0].replace("<", " ").strip()
        given = parts[1].replace("<", " ").strip() if len(parts) > 1 else None
        return surname or None, given or None

    plain_blob = blob.replace("<", "")
    for line in raw_lines:
        candidate = re.sub(r"[^A-Z]", "", line.upper())
        if candidate and plain_blob.startswith(candidate) and len(candidate) < len(plain_blob):
            surname = line.strip()
            remainder = plain_blob[len(candidate):]
            # Find a raw line whose letters match the remainder to recover spacing
            for line2 in raw_lines:
                candidate2 = re.sub(r"[^A-Z]", "", line2.upper())
                if candidate2 == remainder:
                    return surname, line2.strip()
            return surname, remainder  # fallback: no spacing recovered
    return None, None  # couldn't confidently split


def parse_mrz(line1, line2, raw_lines):
    """Parse TD3 (passport) MRZ. Handles variable-length lines (dropped '<')."""
    # ---- Line 1: names ----
    country_code = line1[1:4]
    names_blob = line1[4:]
    surname, given_name = split_name_blob(names_blob, raw_lines)

    # ---- Line 2: fixed-width header (28 chars) is reliable — no filler chars
    # ever appear there for a fully-populated passport, so slicing is safe
    # even if trailing filler elsewhere got dropped. ----
    if len(line2) < 28:
        return {"mrz_checks": {"overall_valid": False}}

    passport_raw = line2[0:9]
    passport_check_raw = line2[9]
    nationality_code = line2[10:13]
    dob_raw = fix_numeric_field(line2[13:19])
    dob_check_raw = fix_numeric_field(line2[19])
    sex_raw = line2[20]
    expiry_raw = fix_numeric_field(line2[21:27])
    expiry_check_raw = fix_numeric_field(line2[27])

    # ---- Tail (personal number + its check digit + composite check digit)
    # is variable-length because dropped '<' shrinks it unpredictably. We
    # don't assume a fixed offset here — instead take the last two captured
    # characters as [personal_check, composite_check] and everything between
    # position 28 and there as the personal number (stripped of any '<'). ----
    tail = line2[28:]
    personal_check_raw = fix_numeric_field(tail[-2]) if len(tail) >= 2 else None
    composite_check_raw = fix_numeric_field(tail[-1]) if len(tail) >= 1 else None
    personal_number_raw = fix_numeric_field(tail[:-2]) if len(tail) > 2 else ""

    passport_number = passport_raw.replace("<", "").strip()
    date_of_birth = parse_mrz_date(dob_raw)
    date_of_expiry = parse_mrz_date(expiry_raw, field_type="expiry")
    personal_number = personal_number_raw.replace("<", "").strip()
    sex = {"M": "M", "F": "F"}.get(sex_raw, None)

    # ---- Checksum validation (only meaningful when digits are where expected) ----
    checks = {
        "passport_number": check_digit(passport_raw) == (int(passport_check_raw) if passport_check_raw.isdigit() else -1),
        "date_of_birth": check_digit(dob_raw) == (int(dob_check_raw) if dob_check_raw.isdigit() else -1),
        "date_of_expiry": check_digit(expiry_raw) == (int(expiry_check_raw) if expiry_check_raw.isdigit() else -1),
    }
    checks["overall_valid"] = checks["passport_number"] and checks["date_of_birth"] and checks["date_of_expiry"]

    return {
        "passport_number": passport_number or None,
        "surname": surname or None,
        "given_name": given_name or None,
        "nationality": NATIONALITY_MAP.get(nationality_code, nationality_code) if nationality_code.strip("<") else None,
        "country_code": country_code if country_code.strip("<") else None,
        "date_of_birth": date_of_birth,
        "sex": sex,
        "date_of_expiry": date_of_expiry,
        "personal_number": personal_number or None,
        "mrz_checks": checks,
    }


# ---------------------------------------------------------------------------
# VIZ (printed text) — fields not present in the MRZ at all: date_of_issue,
# place_of_birth, issuing_authority, previous_passport_number.
#
# OCR frequently garbles these labels beyond simple substring matching (e.g.
# "Surname" -> "Sumame", "Issuing Authority" -> "ssuing Auhorty"), and
# multi-column passport layouts get jumbled when lines are sorted purely by
# y-position. So instead of relying on exact label text, we use structural/
# elimination heuristics wherever possible.
# ---------------------------------------------------------------------------

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

DATE_TEXT_RE = re.compile(r"(\d{1,2})\s*([A-Z]{3})\s*(\d{4})", re.I)
PASSPORT_LIKE_RE = re.compile(r"^[A-Z]{1,2}[0-9]{6,9}$")


def find_printed_dates(lines):
    """Find every date-like token in the printed text, normalized to ISO format,
    keeping the line's y-position so we can order them top-to-bottom."""
    found = []
    for l in lines:
        m = DATE_TEXT_RE.search(l["text"].upper())
        if m:
            day, mon, year = m.groups()
            month = MONTHS.get(mon)
            if month:
                found.append((l["y"], f"{year}-{month}-{day.zfill(2)}"))
    found.sort(key=lambda x: x[0])
    return [d for _, d in found]


def find_date_of_issue(lines, date_of_birth, date_of_expiry):
    """A passport prints exactly 3 dates: DOB, issue, expiry. We already know
    DOB and expiry reliably from the MRZ, so whichever printed date is left
    over (by elimination) must be the issue date. This sidesteps needing to
    match the (often garbled) 'Date of Issue' label at all."""
    dates = find_printed_dates(lines)
    known = {date_of_birth, date_of_expiry}
    for d in dates:
        if d not in known:
            return d
    return None


def find_previous_passport_number(lines, current_passport_number):
    """Look for a passport-number-shaped token (1-2 letters + 6-9 digits)
    that isn't the current passport number."""
    for l in lines:
        token = l["text"].upper().replace(" ", "")
        if PASSPORT_LIKE_RE.match(token) and token != current_passport_number:
            return token
    return None


LABEL_KEYWORDS = {
    "place_of_birth": ["birth"],  # DOB has no separate label in raw OCR here, so
                                    # any 'birth' fragment left over refers to place
    "issuing_authority": ["author", "issuing"],
}


def parse_viz_labels(lines):
    """Best-effort label-then-next-line matching for fields with no more
    reliable extraction path available."""
    texts = [l["text"] for l in lines]
    lower_texts = [t.lower() for t in texts]
    fields = {k: None for k in LABEL_KEYWORDS}

    for field, keywords in LABEL_KEYWORDS.items():
        for i, lt in enumerate(lower_texts):
            if any(kw in lt for kw in keywords):
                if i + 1 < len(texts):
                    fields[field] = texts[i + 1].strip()
                break

    return fields


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    lines = run_raw_ocr(img_array)
    raw_texts = [l["text"] for l in lines]

    mrz = find_mrz_lines(lines)
    mrz_fields = {}
    if mrz:
        mrz_fields = parse_mrz(mrz[0], mrz[1], raw_texts)

    viz_fields = parse_viz_labels(lines)
    date_of_issue = find_date_of_issue(
        lines, mrz_fields.get("date_of_birth"), mrz_fields.get("date_of_expiry")
    )
    previous_passport_number = find_previous_passport_number(
        lines, mrz_fields.get("passport_number")
    )

    response = {
        "passport_number": mrz_fields.get("passport_number"),
        "surname": mrz_fields.get("surname"),
        "given_name": mrz_fields.get("given_name"),
        "nationality": mrz_fields.get("nationality"),
        "sex": mrz_fields.get("sex"),
        "date_of_birth": mrz_fields.get("date_of_birth"),
        "date_of_issue": date_of_issue,
        "date_of_expiry": mrz_fields.get("date_of_expiry"),
        "place_of_birth": viz_fields.get("place_of_birth"),
        "issuing_authority": viz_fields.get("issuing_authority"),
        "personal_number": mrz_fields.get("personal_number"),
        "previous_passport_number": previous_passport_number,
        "_meta": {
            "mrz_detected": mrz is not None,
            "mrz_checks": mrz_fields.get("mrz_checks"),
            "raw_lines": raw_texts,
        },
    }
    return response
