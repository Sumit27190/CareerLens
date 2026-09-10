import json
import os
import re

import faiss
import pymupdf
from sentence_transformers import SentenceTransformer


PDF_PATH = "data/resume.pdf"

VECTOR_STORE_DIR = "vector_store"

INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "resume.index"
)

METADATA_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "metadata.json"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.txt"
)


# ---------------------------------------------------------
# SECTION DETECTION
# ---------------------------------------------------------

SECTION_PATTERNS = {
    "SUMMARY": [
        "professional summary",
        "summary",
    ],

    "SKILLS": [
        "key expertise",
        "skills",
        "technical skills",
    ],

    "EDUCATION": [
        "education",
    ],

    "PROJECTS": [
        "projects",
        "project",
    ],

    "ACHIEVEMENTS": [
        "achievements",
        "achievement",
    ],

    "CERTIFICATIONS": [
        "assessments / certifications",
        "assessments/certifications",
        "certifications",
        "certification",
    ],

    "INTERESTS": [
        "personal interests / hobbies",
        "personal interests/hobbies",
        "interests / hobbies",
        "interests",
        "hobbies",
    ],

    "LINKS": [
        "web links / ims",
        "web links/ims",
        "web links",
        "links",
    ],
}


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize whitespace while preserving the actual content.
    """

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------

def extract_blocks(pdf_path):
    """
    Extract individual text lines from the PDF.

    Font size is preserved because it helps distinguish
    project titles from project descriptions.
    """

    doc = pymupdf.open(pdf_path)

    blocks = []

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        page_data = page.get_text(
            "dict"
        )

        for block in page_data.get(
            "blocks",
            []
        ):

            if "lines" not in block:
                continue

            for line in block["lines"]:

                spans = [
                    span
                    for span in line.get(
                        "spans",
                        []
                    )
                    if span.get(
                        "text",
                        ""
                    ).strip()
                ]

                if not spans:
                    continue

                text = " ".join(
                    span["text"].strip()
                    for span in spans
                )

                text = normalize_text(
                    text
                )

                if not text:
                    continue

                font_size = max(
                    span.get(
                        "size",
                        0
                    )
                    for span in spans
                )

                blocks.append(
                    {
                        "text": text,
                        "font_size": font_size,
                        "page": page_number,
                    }
                )

    doc.close()

    return blocks


# ---------------------------------------------------------
# SECTION IDENTIFICATION
# ---------------------------------------------------------

def get_section_name(text):
    """
    Return the normalized section name if the text
    represents a known resume section.

    Matching is based on section labels, not project names.
    """

    normalized = normalize_text(
        text
    ).lower()

    for section, patterns in SECTION_PATTERNS.items():

        for pattern in patterns:

            if normalized == pattern:
                return section

    return None


# ---------------------------------------------------------
# DATE DETECTION
# ---------------------------------------------------------

def is_date(text):
    """
    Detect common academic/project date formats.
    """

    text = normalize_text(
        text
    )

    if not text:
        return False

    patterns = [
        r"^\d{4}$",

        r"^\d{4}\s*[-–]\s*\d{4}$",

        r"^\d{4}\s*[-–]\s*(present|Present)$",

        r"^\d{1,2}/\d{4}$",

        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",

        r"^\d{4}\s*[-–]\s*$",
    ]

    for pattern in patterns:

        if re.match(
            pattern,
            text
        ):
            return True

    return False


# ---------------------------------------------------------
# PROJECT TITLE DETECTION
# ---------------------------------------------------------

def looks_like_project_title(block):
    """
    Detect project titles from formatting and structure.

    Project names are NOT hardcoded.

    A candidate title should generally be:
    - short
    - non-bullet text
    - not a date
    - not a section heading
    - visually prominent
    """

    text = normalize_text(
        block["text"]
    )

    font_size = block[
        "font_size"
    ]

    if not text:
        return False

    # Never treat section headings as projects.
    if get_section_name(text):
        return False

    # Never treat dates as projects.
    if is_date(text):
        return False

    # Project titles should be reasonably short.
    if len(text) > 80:
        return False

    if len(text.split()) > 12:
        return False

    # Bullet points are descriptions, not titles.
    if text.startswith(
        (
            "•",
            "-",
            "–",
            "*",
        )
    ):
        return False

    # Project titles in the document use larger/
    # prominent formatting.
    if font_size < 9.5:
        return False

    return True


# ---------------------------------------------------------
# GENERAL SECTION CHUNKING
# ---------------------------------------------------------

def create_general_chunk(
    section,
    lines
):
    """
    Create one chunk for a normal resume section.
    """

    if not lines:
        return None

    cleaned_lines = []

    for line in lines:

        line = normalize_text(
            line
        )

        if line:
            cleaned_lines.append(
                line
            )

    if not cleaned_lines:
        return None

    text = "\n".join(
        cleaned_lines
    )

    return {
        "text": text,
        "section": section,
    }


# ---------------------------------------------------------
# PROJECT CHUNKING
# ---------------------------------------------------------

def create_project_chunk(
    project_title,
    project_date,
    project_lines
):
    """
    Create one complete project chunk.

    Project title, date, technologies and description
    stay together.
    """

    if not project_title:
        return None

    parts = [
        project_title
    ]

    if project_date:

        parts.append(
            project_date
        )

    for line in project_lines:

        line = normalize_text(
            line
        )

        if line:

            parts.append(
                line
            )

    text = "\n".join(
        parts
    ).strip()

    if not text:
        return None

    return {
        "text": text,
        "section": "PROJECTS",
        "project": project_title,
    }


# ---------------------------------------------------------
# CREATE SEMANTIC CHUNKS
# ---------------------------------------------------------

def create_chunks(blocks):
    """
    Convert PDF blocks into semantic resume chunks.

    Normal sections:
        one logical chunk

    Projects:
        one chunk per project
    """

    chunks = []

    current_section = "OTHER"

    general_lines = []

    current_project_title = None
    current_project_date = None
    current_project_lines = []

    def flush_general():

        nonlocal general_lines

        chunk = create_general_chunk(
            current_section,
            general_lines
        )

        if chunk:

            chunks.append(
                chunk
            )

        general_lines = []

    def flush_project():

        nonlocal current_project_title
        nonlocal current_project_date
        nonlocal current_project_lines

        chunk = create_project_chunk(
            current_project_title,
            current_project_date,
            current_project_lines
        )

        if chunk:

            chunks.append(
                chunk
            )

        current_project_title = None
        current_project_date = None
        current_project_lines = []

    for block in blocks:

        text = normalize_text(
            block["text"]
        )

        if not text:
            continue

        detected_section = get_section_name(
            text
        )

        # -------------------------------------------------
        # SECTION CHANGE
        # -------------------------------------------------

        if detected_section:

            if current_section == "PROJECTS":

                flush_project()

            else:

                flush_general()

            current_section = (
                detected_section
            )

            continue

        # -------------------------------------------------
        # PROJECT SECTION
        # -------------------------------------------------

        if current_section == "PROJECTS":

            # Detect a new project title.
            if looks_like_project_title(
                block
            ):

                flush_project()

                current_project_title = text

                continue

            # Date belonging to current project.
            if (
                current_project_title
                and is_date(text)
            ):

                current_project_date = text

                continue

            # Remaining content belongs
            # to the current project.
            if current_project_title:

                current_project_lines.append(
                    text
                )

            continue

        # -------------------------------------------------
        # NORMAL SECTION
        # -------------------------------------------------

        general_lines.append(
            text
        )

    # -----------------------------------------------------
    # FLUSH FINAL SECTION
    # -----------------------------------------------------

    if current_section == "PROJECTS":

        flush_project()

    else:

        flush_general()

    return chunks


# ---------------------------------------------------------
# SAVE READABLE CHUNKS
# ---------------------------------------------------------

def save_chunks(chunks):
    """
    Save chunks in a human-readable format.
    """

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for index, chunk in enumerate(
            chunks
        ):

            file.write(
                "=" * 70
            )

            file.write("\n")

            file.write(
                f"CHUNK {index} | "
                f"{chunk['section']}"
            )

            file.write("\n")

            file.write(
                "=" * 70
            )

            file.write("\n")

            file.write(
                chunk["text"]
            )

            file.write("\n\n")


# ---------------------------------------------------------
# CREATE FAISS VECTOR STORE
# ---------------------------------------------------------

def create_vector_store(chunks):
    """
    Generate embeddings and store them in FAISS.
    """

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Generating embeddings..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    # -----------------------------------------------------
    # FAISS INDEX
    # -----------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    metadata = []

    for chunk_id, chunk in enumerate(
        chunks
    ):

        item = {
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "section": chunk["section"],
            "source": "resume.pdf",
        }

        if "project" in chunk:

            item["project"] = (
                chunk["project"]
            )

        metadata.append(
            item
        )

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    return embeddings, index


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not os.path.exists(
        PDF_PATH
    ):

        raise FileNotFoundError(
            f"Resume not found at: "
            f"{PDF_PATH}"
        )

    os.makedirs(
        VECTOR_STORE_DIR,
        exist_ok=True
    )

    print(
        "=" * 70
    )

    print(
        "RESUME INGESTION"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # EXTRACT
    # -----------------------------------------------------

    print(
        "\nReading resume..."
    )

    blocks = extract_blocks(
        PDF_PATH
    )

    print(
        f"Extracted blocks: "
        f"{len(blocks)}"
    )

    # -----------------------------------------------------
    # CHUNK
    # -----------------------------------------------------

    print(
        "\nCreating semantic chunks..."
    )

    chunks = create_chunks(
        blocks
    )

    print(
        f"Created chunks: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    for index, chunk in enumerate(
        chunks
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"CHUNK {index} | "
            f"{chunk['section']}"
        )

        print(
            "=" * 70
        )

        print(
            chunk["text"]
        )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    save_chunks(
        chunks
    )

    # -----------------------------------------------------
    # EMBEDDINGS + FAISS
    # -----------------------------------------------------

    embeddings, index = (
        create_vector_store(
            chunks
        )
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RESUME VECTOR STORE CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Vectors stored: "
        f"{index.ntotal}"
    )

    print(
        f"FAISS index: "
        f"{INDEX_PATH}"
    )

    print(
        f"Metadata: "
        f"{METADATA_PATH}"
    )

    print(
        f"Readable chunks: "
        f"{CHUNKS_PATH}"
    )


if __name__ == "__main__":
    main()