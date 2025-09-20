from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import re
import spacy
from pdf2image import convert_from_bytes
import io
import base64

# --- FIX for NLTK stopwords error ---
import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

from pyresparser import ResumeParser
# from resume_parser import resumeparse

# ------------ Flask ------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------ NLP ------------
nlp = spacy.load("en_core_web_sm")

CITY_TERMS = {
    "ghaziabad", "noida", "delhi", "new delhi", "gurgaon", "gurugram", "faridabad",
    "mumbai", "pune", "bengaluru", "bangalore", "hyderabad", "chennai", "kolkata",
    "ahmedabad", "jaipur", "lucknow", "indore", "thane"
}

DEGREE_PATTERN = re.compile(
    r"\b("
    r"b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|bca|mca|mba|b\.?com|m\.?com|ph\.?d|"
    r"bachelor(?: of)?|master(?: of)?"
    r")\b",
    re.IGNORECASE
)

INSTITUTION_HINT = re.compile(
    r"\b(university|college|institute|school|academy|iit|nit|iiit)\b",
    re.IGNORECASE
)

YEAR_RANGE = re.compile(r"((?:19|20)\d{2})(?:\s*[–—-]\s*((?:19|20)\d{2}))?")
FIELD_HINT = re.compile(
    r"(computer science|information technology|it|electronics(?: and communication)?|ece|electrical|mechanical|civil|"
    r"data science|artificial intelligence|machine learning|business administration|commerce|physics|chemistry|"
    r"mathematics|statistics|economics|management|software engineering)",
    re.IGNORECASE
)

LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub|company)/[A-Za-z0-9\-_/]+",
    re.IGNORECASE
)

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")

def extract_with_pyresparser(pdf_path: str) -> dict:
    try:
        data = ResumeParser(pdf_path).get_extracted_data()
        return data
    except Exception as e:
        print("PyResparser error:", e)
        return {}
    
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    text = "\n".join(parts)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ------------ Field extraction helpers ------------
def extract_name(doc, full_text: str) -> str | None:
    # Prefer short PERSON spans that don't include known city terms
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        candidate = ent.text.strip()
        # Remove any city word even if attached at end (e.g., "Vicky KumarGhaziabad")
        for city in CITY_TERMS:
            candidate = re.sub(city + r"$", "", candidate, flags=re.IGNORECASE).strip()

        # Remove trailing punctuation/garbage
        candidate = candidate.strip(" •|-–—,.;:()[]{}")

        # Heuristic filters
        if any(c in candidate for c in "@0123456789"):
            continue
        tokens = candidate.split()
        if not (1 <= len(tokens) <= 4):
            continue
        if len(candidate) > 40:
            continue

        # Title-case fix
        return " ".join(t.capitalize() for t in tokens)

    # Fallback: first non-empty line that looks like a name
    for line in full_text.splitlines():
        s = line.strip(" •|-–—,.;:()[]{}")
        if 2 <= len(s) <= 40 and s and s[0].isalpha() and "@" not in s and any(ch.isalpha() for ch in s):
            # Avoid lines that are clearly addresses/cities
            if any(city in s.lower() for city in CITY_TERMS):
                continue
            return s.title()
    return None


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    if not m:
        return None
    email = m.group(0).lower()
    return email.strip(" .;,()[]{}<>|")


def extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text)
    if not m:
        return None
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    # Normalize Indian numbers (store 10 or +91XXXXXXXXXX)
    if len(digits) >= 12 and digits.startswith("91"):
        digits = digits[-10:]
        return "+91 " + digits[:5] + " " + digits[5:]
    if len(digits) >= 10:
        digits = digits[-10:]
        return digits[:5] + " " + digits[5:]
    return raw


def extract_linkedin(text: str) -> str | None:
    m = LINKEDIN_RE.search(text)
    if not m:
        return None
    link = m.group(0)
    if not link.lower().startswith("http"):
        link = "https://" + link
    return link


def extract_summary(text: str) -> str | None:
    # Grab content after "Summary" or "Professional Summary"
    pattern = re.compile(r"(professional summary|summary)\s*[:\-]?\s*(.+?)(?:\n\n|experience|education|skills)", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(2)).strip()
    return None


def extract_skills(text: str) -> list[str]:
    keywords = [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "React", "Node.js", "Django", "Flask",
        "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "HTML", "CSS", "Git", "GitHub",
        "AWS", "Docker", "Kubernetes", "REST", "JWT", "Vite"
    ]
    found = set()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            found.add(kw)
    return sorted(found)


# --- Strict, anchored heading matchers ---
EXPERIENCE_HEADING_RE = re.compile(
    r"^\s*((work|professional)\s+)?experience(s)?\s*:?\s*$"
    r"|^\s*employment\s+history\s*:?\s*$"
    r"|^\s*internships?\s*(experience)?\s*:?\s*$",
    re.IGNORECASE,
)

SECTION_STOP_RE = re.compile(
    r"^\s*(education|skills?|projects?|certifications?|achievements?|awards?|publications|"
    r"activities|volunteer(ing)?|extras?)\s*:?\s*$",
    re.IGNORECASE,
)

MONTHS_RE = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|" \
            r"Sep(?:t(?:ember)?)?\.?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"

YEAR_RANGE_FLEX = re.compile(
    r"((?:19|20)\d{2})(?:.{0,40}?[–—-].{0,40}?((?:19|20)\d{2}|Present))?",
    re.IGNORECASE,
)

YEAR_SPAN = re.compile(  # tight 2-capture range like "2021 – 2024" / "2022 – Present"
    r"((?:19|20)\d{2}).{0,40}?(?:–|—|-).{0,40}?((?:19|20)\d{2}|Present)",
    re.IGNORECASE
)


# FIELD_HINT = re.compile(
#     r"(computer science|information technology|it|electronics(?: and communication)?|ece|electrical|mechanical|civil|"
#     r"data science|artificial intelligence|machine learning|business administration|commerce|physics|chemistry|"
#     r"mathematics|statistics|economics|management|software engineering|computer applications|bca|mca|"
#     r"law|medicine|architecture|fashion design|psychology|sociology|political science|history|english literature)"
#     , re.IGNORECASE
# )

FIELD_HINT = re.compile(
    r"(computer\s+applications?|information\s+technology|computer\s+science|"
    r"electronics|mechanical|civil|electrical|commerce|business\s+administration|"
    r"arts|science|law|medicine|data\s+science|ai|ml|physics|chemistry|mathematics)",
    re.IGNORECASE
)


DEGREE_PATTERN = re.compile(
    r"\b("
    # Prefer long forms first to avoid partial generic 'bachelor'
    r"bachelor\s+of\s+computer\s+applications?|"     # Bachelor of Computer Application(s)
    r"master\s+of\s+computer\s+applications?|"       # Master of Computer Application(s)

    # Acronyms with dots/spaces allowed
    r"b\.?\s*c\.?\s*a|m\.?\s*c\.?\s*a|"              # BCA / MCA (B.C.A, B C A, etc.)

    # Other common degrees (kept as you had them)
    r"b\.?\s*tech|m\.?\s*tech|b\.?\s*e|m\.?\s*e|b\.?\s*sc|m\.?\s*sc|ba|ma|"
    r"mba|b\.?\s*com|m\.?\s*com|bba|bbm|bms|bjmc|bmm|bj|mj|bva|bvoc|"
    r"bpharm|mpharm|mbbs|bds|bhms|bams|bpt|mpt|"
    r"iti|polytechnic|pgdm|pgdca|diploma|certificate|"
    r"ph\.?\s*d|llb|llm|b\.?\s*ed|m\.?\s*ed|"

    # Generic fallbacks (last, to avoid swallowing specifics)
    r"bachelor(?:'s| of| in)?|master(?:'s| of| in)?|associate(?:'s| of| in)?"
    r")\b",
    re.IGNORECASE
)

INSTITUTION_HINT = re.compile(
    r"\b(university|college|institute|school|academy|campus|iit|nit|iiit|ggsipu|ipu|aktu|univ)\b",
    re.IGNORECASE
)

def parse_education(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # --- Find Education heading ---
    edu_heading = re.compile(r"^\s*education\s*:?\s*$", re.IGNORECASE)
    start = next((i for i, ln in enumerate(lines) if edu_heading.match(ln)), None)
    if start is None:
        return [], None

    # --- Section end ---
    end = next(
        (i for i, ln in enumerate(lines[start + 1:], start + 1) if SECTION_STOP_RE.match(ln)),
        len(lines),
    )
    block = lines[start + 1:end]

    # 🔥 Merge lines into one string for parsing
    chunk = " ".join(block)

    degree, field, year_str, institution = "", "", "", ""

    # --- DEGREE detection ---
    m = DEGREE_PATTERN.search(chunk)
    if m:
        degree = re.sub(r"\s+", " ", m.group(0)).strip().title()
        field_text = chunk[m.end():]
        fm = FIELD_HINT.search(field_text)
        if fm:
            field = fm.group(0).title()
        elif "Application" in degree:
            field = "Computer Application"

    # --- YEAR detection ---
    y_m = YEAR_RANGE_FLEX.search(chunk)
    if y_m:
        year_str = "–".join([p for p in y_m.groups() if p])

    # --- INSTITUTION detection ---
    for ln in block:
        if INSTITUTION_HINT.search(ln):
            institution = ln
            break

    entry = {
        "degree": degree,
        "field": field,
        "institution": institution,
        "year": year_str,
    }

    if any(entry.values()):
        return [entry], None
    return [], "\n".join(block)

def parse_experience(text: str) -> tuple[list[dict], str | None]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    trimmed = [ln.strip() for ln in lines]

    start = next((i for i, ln in enumerate(trimmed) if EXPERIENCE_HEADING_RE.match(ln)), None)
    if start is None:
        return [], None

    end = next((i for i, ln in enumerate(trimmed[start + 1:], start + 1) if SECTION_STOP_RE.match(ln)), len(trimmed))
    block = [ln.strip() for ln in lines[start + 1:end] if ln.strip()]

    # Entry starts: a line with a year that is NOT a bullet
    year_re = re.compile(r"(?:19|20)\d{2}")
    bullet_re = re.compile(r"^[•\-–]")
    starts = [i for i, ln in enumerate(block) if year_re.search(ln) and not bullet_re.match(ln)]
    if not starts:
        return [{
            "title": "", "company": "", "dates": "", "location": "",
            "description": " ".join(block).strip()
        }], None
    starts.append(len(block))

    results = []
    for si, ei in zip(starts, starts[1:]):
        chunk = block[si:ei]
        header1 = chunk[0] if chunk else ""
        header2 = chunk[1] if len(chunk) > 1 else ""

        # Dates (handles "Sept. 2023 – Oct. 2023", "2022 – Present", etc.)
        m = YEAR_RANGE_FLEX.search(header1) or YEAR_RANGE_FLEX.search(header2)
        dates = "–".join([p for p in (m.groups() if m else []) if p]) if m else ""

        # Company = part of header1 before the first year, with trailing month stripped
        first_year = year_re.search(header1)
        comp_part = header1[:first_year.start()] if first_year else header1
        comp_part = re.sub(rf"{MONTHS_RE}\s*$", "", comp_part, flags=re.IGNORECASE).strip(" -–—,.;:")
        company = comp_part.strip()

        # Title + Location (detects city at end)
        hdr = header2 or (header1[first_year.end():] if first_year else header2)
        title, location = "", ""
        if hdr:
            toks = hdr.split()
            if toks and toks[-1].lower() in CITY_TERMS:
                location = toks[-1].title()
                title = " ".join(toks[:-1]).strip()
            else:
                title = hdr.strip()

        description = " ".join(chunk[2:]).strip() if len(chunk) > 2 else ""
        results.append({"title": title, "company": company, "dates": dates, "location": location, "description": description})

    return results, None

def extract_text_fields(pdf_path: str) -> dict:
    try:
        all_text = read_pdf_text(pdf_path)
        doc = nlp(all_text)

        from_regex = {
            "name": extract_name(doc, all_text),
            "email": extract_email(all_text),
            "phone": extract_phone(all_text),
            "linkedin": extract_linkedin(all_text),
            "summary": extract_summary(all_text),
            "skills": ", ".join(extract_skills(all_text)),
        }

        edu_list, edu_raw = parse_education(all_text)
        exp_list, exp_raw = parse_experience(all_text)
        from_regex["education_list"] = edu_list or []
        from_regex["experience_list"] = exp_list or []
        return from_regex
    except Exception as e:
        print(f"Regex extraction failed: {e}")
        return {}

def extract_image_from_pdf(pdf_path: str) -> str | None:
    try:
        with open(pdf_path, "rb") as f:
            images = convert_from_bytes(f.read())
        if images:
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format="PNG")
            return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Image extraction failed: {e}")
    return None

# ------------ Routes ------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(file_path)

        parsed = {}

        # ---------- STEP 1: Try PyResparser ----------
        try:
            parsed = ResumeParser(file_path).get_extracted_data() or {}
        except Exception as e:
            print("PyResparser failed:", e)

        # ---------- STEP 2: Fallback to Regex ----------
        if not parsed or not parsed.get("name"):
            parsed = extract_text_fields(file_path) or {}

        # ---------- Normalize Keys ----------
        fields = {
            "name": parsed.get("name") or parsed.get("Name"),
            "email": parsed.get("email") or parsed.get("Email"),
            "phone": parsed.get("phone") or parsed.get("Mobile") or parsed.get("contact"),
            "linkedin": parsed.get("linkedin") or parsed.get("LinkedIn"),
            "summary": parsed.get("summary") or "",
            "skills": (
                ", ".join(parsed.get("skills", []))
                if isinstance(parsed.get("skills"), list)
                else parsed.get("skills", "")
            ),
            "education_list": parsed.get("education_list") or parsed.get("education", []) or [],
            "experience_list": parsed.get("experience_list") or parsed.get("experience", []) or []
        }

        # ---------- Image Extraction ----------
        image_data = extract_image_from_pdf(file_path)

        return jsonify({
            "success": True,
            "fields": fields,
            "has_image": bool(image_data),
            "image": image_data,
            "manual_image_required": not bool(image_data),
        })

    except Exception as e:
        app.logger.exception("Upload failed")
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
