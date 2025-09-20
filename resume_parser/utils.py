from pdf2image import convert_from_bytes
import io, base64

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
