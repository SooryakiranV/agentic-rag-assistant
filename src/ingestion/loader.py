import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dictionaries containing page number and extracted text.
    """
    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append(
            {
                "page_number": page_number,
                "text": text.strip(),
            }
        )

    document.close()

    return pages