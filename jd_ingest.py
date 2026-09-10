import json
import os
import re

import faiss
from sentence_transformers import SentenceTransformer


JD_PATH = "data/job_description.txt"

VECTOR_STORE_DIR = "vector_store"

INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "jd.index"
)

METADATA_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "jd_metadata.json"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "jd_chunks.txt"
)


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize whitespace and remove unnecessary formatting
    characters while preserving the actual content.
    """

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_heading(text):
    """
    Normalize a possible heading so that variations such as:

        What you'll get to do...
        What you'll get to do...•
        What you'll get to do... •

    are treated as the same heading.
    """

    text = normalize_text(
        text
    )

    # Remove bullet characters.
    text = re.sub(
        r"[•●▪◦]+",
        "",
        text
    )

    # Remove trailing punctuation / dots.
    text = re.sub(
        r"[\s.!:;]+$",
        "",
        text
    )

    # Collapse whitespace again.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip().lower()


# ---------------------------------------------------------
# HEADING DETECTION
# ---------------------------------------------------------

def is_heading(line):
    """
    Determine whether a line is a JD section heading.

    The function uses structural patterns rather than
    individual job requirements.
    """

    normalized = normalize_heading(
        line
    )

    if not normalized:
        return False

    # Headings should be reasonably short.
    if len(normalized) > 100:
        return False

    # -----------------------------------------------------
    # Generic heading patterns
    # -----------------------------------------------------

    heading_patterns = [
        r"^what\s+you(?:'ll|’ll|\s+will)\s+get\s+to\s+do$",

        r"^your\s+experience\s+should\s+include$",

        r"^you\s+might\s+also\s+have$",

        r"^about\s+the\s+role$",

        r"^about\s+the\s+team$",

        r"^about\s+us$",

        r"^requirements$",

        r"^qualifications$",

        r"^responsibilities$",

        r"^preferred\s+qualifications$",

        r"^basic\s+qualifications$",

        r"^preferred$",

        r"^responsibilities\s+include$",

        r"^what\s+we\s+offer$",

        r"^what\s+we\s+do$",
    ]

    for pattern in heading_patterns:

        if re.match(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):
            return True

    return False


# ---------------------------------------------------------
# HEADING NAME
# ---------------------------------------------------------

def get_heading_name(line):
    """
    Return a clean display name for a detected heading.
    """

    normalized = normalize_heading(
        line
    )

    heading_map = {
        "what you'll get to do":
            "What you'll get to do",

        "what you’ll get to do":
            "What you'll get to do",

        "what you will get to do":
            "What you'll get to do",

        "your experience should include":
            "Your experience should include",

        "you might also have":
            "You might also have",

        "about the role":
            "About the role",

        "about the team":
            "About the team",

        "about us":
            "About us",

        "requirements":
            "Requirements",

        "qualifications":
            "Qualifications",

        "responsibilities":
            "Responsibilities",

        "preferred qualifications":
            "Preferred qualifications",

        "basic qualifications":
            "Basic qualifications",

        "preferred":
            "Preferred",

        "responsibilities include":
            "Responsibilities include",

        "what we offer":
            "What we offer",

        "what we do":
            "What we do",
    }

    return heading_map.get(
        normalized,
        normalize_text(line)
    )


# ---------------------------------------------------------
# SPLIT LONG TEXT
# ---------------------------------------------------------

def split_long_text(
    text,
    max_chars=700
):
    """
    Split long text into chunks while trying to preserve
    complete sentences.
    """

    text = normalize_text(
        text
    )

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current_chunk = ""

    for sentence in sentences:

        sentence = normalize_text(
            sentence
        )

        if not sentence:
            continue

        if not current_chunk:

            current_chunk = sentence

            continue

        candidate = (
            current_chunk
            + " "
            + sentence
        )

        if len(candidate) <= max_chars:

            current_chunk = candidate

        else:

            chunks.append(
                current_chunk
            )

            current_chunk = sentence

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks


# ---------------------------------------------------------
# CREATE SECTIONS
# ---------------------------------------------------------

def create_sections(text):
    """
    Divide the JD into logical sections based on headings.
    """

    lines = [
        normalize_text(line)
        for line in text.splitlines()
    ]

    lines = [
        line
        for line in lines
        if line
    ]

    sections = []

    current_section = "GENERAL"

    current_lines = []

    for line in lines:

        if is_heading(line):

            # Save content collected before this heading.
            if current_lines:

                section_text = " ".join(
                    current_lines
                )

                sections.append(
                    {
                        "section": current_section,
                        "text": section_text
                    }
                )

            current_section = get_heading_name(
                line
            )

            current_lines = []

            continue

        current_lines.append(
            line
        )

    # Save final section.
    if current_lines:

        section_text = " ".join(
            current_lines
        )

        sections.append(
            {
                "section": current_section,
                "text": section_text
            }
        )

    return sections


# ---------------------------------------------------------
# CREATE CHUNKS
# ---------------------------------------------------------

def create_chunks(text):
    """
    Convert JD sections into embedding chunks.

    Sections are preserved as metadata.

    Long sections are split into smaller chunks.
    """

    sections = create_sections(
        text
    )

    chunks = []

    for section in sections:

        section_text = normalize_text(
            section["text"]
        )

        if not section_text:
            continue

        section_chunks = split_long_text(
            section_text,
            max_chars=700
        )

        for chunk_text in section_chunks:

            chunks.append(
                {
                    "text": chunk_text,
                    "section": section[
                        "section"
                    ]
                }
            )

    return chunks


# ---------------------------------------------------------
# SAVE CHUNKS
# ---------------------------------------------------------

def save_chunks(chunks):
    """
    Save readable chunks for debugging and inspection.
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
# CREATE VECTOR STORE
# ---------------------------------------------------------

def create_vector_store(chunks):
    """
    Generate embeddings for JD chunks and store them
    in a FAISS vector index.
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
        "Generating JD embeddings..."
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
    # FAISS
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

        metadata.append(
            {
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "section": chunk["section"],
                "source": "job_description.txt"
            }
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
        JD_PATH
    ):

        raise FileNotFoundError(
            f"Job description not found: "
            f"{JD_PATH}"
        )

    os.makedirs(
        VECTOR_STORE_DIR,
        exist_ok=True
    )

    print(
        "=" * 70
    )

    print(
        "JOB DESCRIPTION INGESTION"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # READ JD
    # -----------------------------------------------------

    print(
        "\nReading job description..."
    )

    with open(
        JD_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    print(
        f"JD characters: "
        f"{len(text)}"
    )

    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    print(
        "\nCreating semantic chunks..."
    )

    chunks = create_chunks(
        text
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
    # SAVE READABLE CHUNKS
    # -----------------------------------------------------

    save_chunks(
        chunks
    )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS + FAISS
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
        "JD VECTOR STORE CREATED"
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