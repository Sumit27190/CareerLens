import json

import faiss
from sentence_transformers import SentenceTransformer


class ResumeRetriever:

    def __init__(
        self,
        index_path="vector_store/resume.index",
        metadata_path="vector_store/metadata.json",
        model_name="all-MiniLM-L6-v2"
    ):
        """
        Initialize the resume retriever.

        Loads:
        1. The embedding model
        2. The FAISS vector index
        3. The metadata associated with each vector
        """

        self.model = SentenceTransformer(
            model_name
        )

        self.index = faiss.read_index(
            index_path
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.metadata = json.load(
                file
            )

    # -----------------------------------------------------
    # SEMANTIC SEARCH
    # -----------------------------------------------------

    def search(
        self,
        query,
        top_k=3
    ):
        """
        Perform semantic similarity search.

        Steps:

        query
            ↓
        embedding
            ↓
        FAISS search
            ↓
        most relevant resume chunks
        """

        if not query or not query.strip():

            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        # Avoid asking FAISS for more vectors
        # than actually exist.
        top_k = min(
            top_k,
            self.index.ntotal
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            # FAISS can return -1 when there is
            # no valid result.
            if index < 0:
                continue

            result = self.metadata[
                index
            ].copy()

            result["score"] = float(
                score
            )

            results.append(
                result
            )

        return results

    # -----------------------------------------------------
    # STRUCTURED RETRIEVAL
    # -----------------------------------------------------

    def get_all_metadata(self):
        """
        Return all resume metadata.
        """

        return self.metadata

    def get_all_projects(self):
        """
        Return all project chunks.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "PROJECTS"
        ]

    def get_education(self):
        """
        Return education chunks.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "EDUCATION"
        ]

    def get_skills(self):
        """
        Return skill-related chunks.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "SKILLS"
        ]

    def get_achievements(self):
        """
        Return achievement chunks.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "ACHIEVEMENTS"
        ]

    def get_certifications(self):
        """
        Return certification chunks.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "CERTIFICATIONS"
        ]

    def get_summary(self):
        """
        Return the professional summary.
        """

        return [
            item
            for item in self.metadata
            if item.get("section") == "SUMMARY"
        ]

    # -----------------------------------------------------
    # STRUCTURED QUERY ROUTING
    # -----------------------------------------------------

    def structured_search(
        self,
        query
    ):
        """
        Detect questions that require exact structured
        resume information instead of semantic similarity.

        Returns:
            list of results if the query is structured

            None if semantic search should be used
        """

        query_lower = query.lower().strip()

        # -------------------------------------------------
        # PROJECT QUESTIONS
        # -------------------------------------------------

        if (
            "how many projects" in query_lower
            or "number of projects" in query_lower
            or "projects did i build" in query_lower
            or "projects have i built" in query_lower
        ):

            return self.get_all_projects()

        if (
            "list my projects" in query_lower
            or "list projects" in query_lower
            or "what are my projects" in query_lower
            or "what projects did i build" in query_lower
        ):

            return self.get_all_projects()

        # -------------------------------------------------
        # EDUCATION QUESTIONS
        # -------------------------------------------------

        if (
            "education" in query_lower
            or "degree" in query_lower
            or "college" in query_lower
            or "university" in query_lower
            or "cgpa" in query_lower
        ):

            return self.get_education()

        # -------------------------------------------------
        # ACHIEVEMENT QUESTIONS
        # -------------------------------------------------

        if (
            "achievement" in query_lower
            or "achievements" in query_lower
            or "award" in query_lower
            or "competition" in query_lower
        ):

            return self.get_achievements()

        # -------------------------------------------------
        # SKILL QUESTIONS
        # -------------------------------------------------

        if (
            "my skills" in query_lower
            or "my technical skills" in query_lower
            or "technologies do i know" in query_lower
            or "tech stack" in query_lower
        ):

            return self.get_skills()

        # -------------------------------------------------
        # CERTIFICATION QUESTIONS
        # -------------------------------------------------

        if (
            "certification" in query_lower
            or "certifications" in query_lower
            or "certificate" in query_lower
        ):

            return self.get_certifications()

        # -------------------------------------------------
        # SUMMARY QUESTIONS
        # -------------------------------------------------

        if (
            "tell me about myself" in query_lower
            or "professional summary" in query_lower
            or "my summary" in query_lower
        ):

            return self.get_summary()

        # No structured route found.
        return None

    # -----------------------------------------------------
    # HYBRID RETRIEVAL
    # -----------------------------------------------------

    def retrieve(
        self,
        query,
        top_k=3
    ):
        """
        Hybrid retrieval.

        First checks whether the query can be answered
        using structured resume information.

        If not, it performs semantic vector search.
        """

        structured_results = (
            self.structured_search(
                query
            )
        )

        if structured_results is not None:

            return {
                "type": "structured",
                "results": structured_results
            }

        semantic_results = self.search(
            query,
            top_k
        )

        return {
            "type": "semantic",
            "results": semantic_results
        }