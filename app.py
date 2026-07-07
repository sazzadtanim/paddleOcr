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
# MRZ detection + OCR-confusion correction
# ---------------------------------------------------------------------------

MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{30,44}$")

# Common OCR misreads when the expected character is a digit
DIGIT_CONFUSION = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}


def fix_numeric_field(s):
    """Correct common letter/digit OCR confusions in a field expected to be all digits (or '<')."""
    return "".join(DIGIT_CONFUSION.get(c, c) if c not in "0123456789<" else c for c in s)


def find_mrz_lines(lines):
    """Return the two MRZ lines (TD3 passport format) if found, else None."""
    candidates = [l["text"].upper().replace(" ", "") for l in lines]
    for i in range(len(candidates) - 1):
        l1, l2 = candidates[i], candidates[i + 1]
        if l1.startswith("P<") and MRZ_LINE_RE.match(l1) and MRZ_LINE_RE.match(l2):
            return l1.ljust(44, "<")[:44], l2.ljust(44, "<")[:44]
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


def parse_mrz(line1, line2):
    """Parse TD3 (passport) MRZ with OCR-confusion correction and checksum validation."""
    # ---- Line 1: names (alpha field, no numeric correction needed) ----
    names_part = line1[5:]
    name_fields = names_part.split("<<", 1)
    surname = name_fields[0].replace("<", " ").strip() if name_fields else None
    given_name = name_fields[1].replace("<", " ").strip() if len(name_fields) > 1 else None
    country_code = line1[2:5]

    # ---- Line 2: fix numeric-only zones before parsing ----
    passport_raw = line2[0:9]
    passport_check_raw = line2[9]
    nationality_code = line2[10:13]
    dob_raw = fix_numeric_field(line2[13:19])
    dob_check_raw = fix_numeric_field(line2[19])
    sex_raw = line2[20]
    expiry_raw = fix_numeric_field(line2[21:27])
    expiry_check_raw = fix_numeric_field(line2[27])
    personal_number_raw = fix_numeric_field(line2[28:42])
    personal_check_raw = fix_numeric_field(line2[42])
    composite_check_raw = fix_numeric_field(line2[43])

    passport_number = passport_raw.replace("<", "").strip()
    date_of_birth = parse_mrz_date(dob_raw)
    date_of_expiry = parse_mrz_date(expiry_raw, field_type="expiry")
    personal_number = personal_number_raw.replace("<", "").strip()
    sex = {"M": "M", "F": "F"}.get(sex_raw, None)

    # ---- Checksum validation ----
    checks = {
        "passport_number": check_digit(passport_raw) == (int(passport_check_raw) if passport_check_raw.isdigit() else -1),
        "date_of_birth": check_digit(dob_raw) == (int(dob_check_raw) if dob_check_raw.isdigit() else -1),
        "date_of_expiry": check_digit(expiry_raw) == (int(expiry_check_raw) if expiry_check_raw.isdigit() else -1),
        "personal_number": check_digit(personal_number_raw) == (int(personal_check_raw) if personal_check_raw.isdigit() else -1),
    }
    composite_input = line2[0:10] + line2[13:20] + line2[21:43]
    composite_input = fix_numeric_field(composite_input)
    checks["composite"] = check_digit(composite_input) == (int(composite_check_raw) if composite_check_raw.isdigit() else -1)
    checks["overall_valid"] = all(checks.values())

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
# VIZ (printed text) label matching — for fields not present in the MRZ
# ---------------------------------------------------------------------------

MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

DATE_TEXT_RE = re.compile(r"(\d{1,2})\s*([A-Z]{3})\s*(\d{4})", re.I)


def normalize_viz_date(text):
    m = DATE_TEXT_RE.search(text.upper())
    if not m:
        return None
    day, mon, year = m.groups()
    month = MONTHS.get(mon)
    if not month:
        return None
    return f"{year}-{month}-{day.zfill(2)}"


LABEL_KEYWORDS = {
    "date_of_issue": ["date of issue"],
    "place_of_birth": ["place of birth"],
    "issuing_authority": ["issuing authority"],
    "previous_passport_number": ["previous passport"],
    "type": ["type"],
}


def parse_viz_labels(lines):
    """Scan OCR'd printed lines: when a label line is found, take the following
    line as its value (matches the label-above-value layout used on passports)."""
    texts = [l["text"] for l in lines]
    lower_texts = [t.lower() for t in texts]
    fields = {k: None for k in LABEL_KEYWORDS}

    for field, keywords in LABEL_KEYWORDS.items():
        for i, lt in enumerate(lower_texts):
            if any(kw in lt for kw in keywords):
                if i + 1 < len(texts):
                    value = texts[i + 1].strip()
                    fields[field] = value
                break

    if fields.get("date_of_issue"):
        normalized = normalize_viz_date(fields["date_of_issue"])
        if normalized:
            fields["date_of_issue"] = normalized

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

    mrz = find_mrz_lines(lines)
    mrz_fields = {}
    if mrz:
        mrz_fields = parse_mrz(mrz[0], mrz[1])

    viz_fields = parse_viz_labels(lines)

    response = {
        "passport_number": mrz_fields.get("passport_number"),
        "surname": mrz_fields.get("surname"),
        "given_name": mrz_fields.get("given_name"),
        "nationality": mrz_fields.get("nationality"),
        "sex": mrz_fields.get("sex"),
        "date_of_birth": mrz_fields.get("date_of_birth"),
        "date_of_issue": viz_fields.get("date_of_issue"),
        "date_of_expiry": mrz_fields.get("date_of_expiry"),
        "place_of_birth": viz_fields.get("place_of_birth"),
        "issuing_authority": viz_fields.get("issuing_authority"),
        "personal_number": mrz_fields.get("personal_number"),
        "previous_passport_number": viz_fields.get("previous_passport_number"),
        "_meta": {
            "mrz_detected": mrz is not None,
            "mrz_checks": mrz_fields.get("mrz_checks"),
            "raw_lines": [l["text"] for l in lines],
        },
    }
    return response
