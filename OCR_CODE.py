# API_KEY = os.getenv("MISTRAL_API_KEY", "7evfwpu0lTwDJYlZ7eFp54PgHhvjBehX")
import sys
import csv
import json
import os
from pathlib import Path

from mistralai import Mistral, UserMessage

# --- Configuration ---
API_KEY = os.getenv("MISTRAL_API_KEY", "7evfwpu0lTwDJYlZ7eFp54PgHhvjBehX")
PDF_FILE_PATH = "/home/polina/PycharmProjects/War_Poetry/Poetic Speech on War: Z_Poetry/Победа_будет_за_нами.pdf"
CSV_OUTPUT_PATH = "/home/polina/PycharmProjects/War_Poetry/Poetic Speech on War: Z_Poetry/poems_extracted_llm_parsed/Pobeda_budet_za_namy.csv"
OCR_MODEL = "mistral-ocr-latest"
PARSING_MODEL = "mistral-large-latest"
PAGES_TO_PROCESS = 20  # Uncomment to limit pages

print(f"Using Python executable: {sys.executable}")
print(f"OCR model: {OCR_MODEL}, parsing model: {PARSING_MODEL}")
print(f"PDF input: {PDF_FILE_PATH}")
print(f"CSV output: {CSV_OUTPUT_PATH}")
print("-" * 40)

# --- Initialize client ---
client = Mistral(api_key=API_KEY)
print("✅ Mistral client initialized.")
print("-" * 40)

# --- Verify PDF file exists ---
pdf_path = Path(PDF_FILE_PATH)
if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF not found at {PDF_FILE_PATH}")
print(f"✅ Found PDF: {pdf_path.name}")
print("-" * 40)

# --- Upload PDF for OCR ---
print("Uploading PDF for OCR…")
upload_resp = client.files.upload(
    file={"file_name": pdf_path.name, "content": pdf_path.read_bytes()},
    purpose="ocr"
)
file_id = upload_resp.id
print(f"✅ Uploaded file ID: {file_id}")

# --- Get signed URL ---
signed = client.files.get_signed_url(file_id=file_id)
signed_url = signed.url
print(f"✅ Signed URL: {signed_url[:50]}…")
print("-" * 40)

# --- Run OCR ---
print("Starting OCR… this may take a while.")
ocr_resp = client.ocr.process(
    model=OCR_MODEL,
    document={"type": "document_url", "document_url": signed_url},
    include_image_base64=False
)
ocr_data = ocr_resp.model_dump()
print(f"✅ OCR complete: found {len(ocr_data.get('pages', []))} pages")
print("-" * 40)

# Clean up upload
client.files.delete(file_id=file_id)
print("🔄 Uploaded file deleted from server")
print("-" * 40)

# --- Parse pages with LLM and filter ---
all_poems = []
pages = ocr_data.get("pages", [])
num_pages = min(PAGES_TO_PROCESS, len(pages))  # Use if limiting pages
#num_pages = len(pages)
print(f"Parsing {num_pages} pages…")

last_author = None
skip_keywords = ["Содержание", "ISBN", "©", "Антология", "Предисловие"]

for idx in range(num_pages):
    page_num = idx + 1
    page = pages[idx]

    # Extract text field
    text = page.get("markdown") or page.get("text") or "\n".join(
        blk.get("text", "") for blk in page.get("blocks", [])
    )
    if not text.strip():
        continue

    # Build parsing prompt
    prompt = f"""
Analyze the following text extracted from a page of a Russian poetry anthology:

--- OCR TEXT START ---
{text}
--- OCR TEXT END ---

Extract all distinct poems, each as an object with:
- author (string; use \"Unknown\" if missing)
- title_or_first_line (string; if missing or placeholders, substitute)
- text (string with line/stanza breaks preserved)

Ignore non-poem content (title pages, TOC, page numbers, headers, epigraphs).
Output strictly a JSON list of these objects.
"""

    resp = client.chat.complete(
        model=PARSING_MODEL,
        messages=[UserMessage(content=prompt)],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw = resp.choices[0].message.content or ""

    # Try parsing JSON
    try:
        parsed_data = json.loads(raw)
        # Ensure we have a list of poems in the expected format
        if isinstance(parsed_data, dict) and "poems" in parsed_data:
            poems = parsed_data["poems"]
        else:
            poems = parsed_data
        
        if not isinstance(poems, list):
            print(f"❌ Unexpected data format on page {page_num}, skipping.")
            continue
            
    except json.JSONDecodeError:
        print(f"❌ JSON parse error on page {page_num}, skipping.")
        continue

    # Process each poem
    for item in poems:
        if not isinstance(item, dict):
            print(f"❌ Skipping invalid poem entry on page {page_num}")
            continue
            
        # Author propagation
        author_raw = str(item.get("author", "")).strip()
        if author_raw and author_raw != "Unknown":
            last_author = author_raw
        author = last_author or "Unknown"

        poem_text = item.get("text", "").strip()
        # Skip short or non-poem items
        lines = [l for l in poem_text.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        # Skip by keywords
        if any(k in poem_text for k in skip_keywords):
            continue

        # Title handling: placeholders '*' or '***' trigger fallback
        title_raw = item.get("title_or_first_line", "").strip()
        if title_raw in ("***", "*"):
            words = poem_text.split()
            title = " ".join(words[:6])
        else:
            title = title_raw
        # Safety net: if title still empty, fallback
        if not title:
            words = poem_text.split()
            title = " ".join(words[:6])

        # Append result
        all_poems.append({
            "author": author,
            "title": title,
            "text": poem_text
        })

    print(f"→ Page {page_num}: items={len(poems)}, total_kept={len(all_poems)}")

print("-" * 40)
print(f"Total poems after filtering: {len(all_poems)}")

# --- Write to CSV ---
with open(CSV_OUTPUT_PATH, "w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=["author", "title", "text"])
    writer.writeheader()
    for poem in all_poems:
        writer.writerow(poem)

print(f"✅ Wrote {len(all_poems)} poems to {CSV_OUTPUT_PATH}")
print("🎉 Done!")