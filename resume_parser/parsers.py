from pyresparser import ResumeParser
# from resume_parser import resumeparse  # optional second library
from resume_parser.extractors import extract_text_fields

def parse_resume(pdf_path: str) -> dict:
    parsed = {}

    # ---------- STEP 1: Try PyResparser ----------
    try:
        parsed = ResumeParser(pdf_path).get_extracted_data() or {}
    except Exception as e:
        print("PyResparser failed:", e)

    # ---------- STEP 2: Regex Fallback ----------
    if not parsed or not parsed.get("name"):
        parsed = extract_text_fields(pdf_path) or {}

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

    return fields
