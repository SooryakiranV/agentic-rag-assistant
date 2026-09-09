from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split page text into overlapping chunks while preserving page metadata.

    Args:
        pages: List of page dictionaries containing page number and text.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of dictionaries containing chunk ID, page number, and text.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_index, text in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"page_{page['page_number']}_chunk_{chunk_index}",
                    "page_number": page["page_number"],
                    "text": text,
                }
            )

    return chunks