
from pypdf import PdfReader
import io

def extract_text_from_pdf(file):
        # Ensure a seekable BytesIO stream
    text = ""
    try:
        file.seek(0)
        pdf_stream = io.BytesIO(file.read())

        pdf_reader = PdfReader(pdf_stream)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as e:
        text = f"Error reading PDF: {e}"

    return text
