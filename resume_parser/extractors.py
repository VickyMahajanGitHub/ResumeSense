import re
import spacy
from PyPDF2 import PdfReader

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# --- Regex constants (reuse from your current code) ---
CITY_TERMS = {"ghaziabad","noida","delhi","gurgaon","mumbai","pune","bengaluru","hyderabad","chennai","kolkata"}
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
# ... (keep DEGREE_PATTERN, FIELD_HINT, LINKEDIN_RE, etc. from your current app.py)

# --- PDF Reader ---
def read_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)

# --- Extraction helpers (examples, move all from your app.py) ---
def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    return m.group(0).lower() if m else None

def extract_phone(text: str) -> str | None:
    m = PHONE_RE.search(text)
    return m.group(0) if m else None

# TODO: move extract_name, extract_summary, extract_skills, parse_education, parse_experience here

def extract_text_fields(pdf_path: str) -> dict:
    all_text = read_pdf_text(pdf_path)
    doc = nlp(all_text)

    return {
        "name": None,  # use extract_name(doc, all_text)
        "email": extract_email(all_text),
        "phone": extract_phone(all_text),
        "linkedin": None,  # use extract_linkedin
        "summary": None,   # use extract_summary
        "skills": [],      # use extract_skills
        "education_list": [],  # use parse_education
        "experience_list": []  # use parse_experience
    }
