import json
import os
import re

from dotenv import load_dotenv
from google import genai

from rag.retrieval import ResumeRetriever


class ResumeMatcher:

    def __init__(
        self,
        jd_metadata_path="vector_store/jd_metadata.json",
        model_name=None
    ):
        """
        Initialize the resume matcher.

        The matcher uses:
        1. JD chunks generated during ingestion
        2. Resume semantic retrieval
        3. Gemini for requirement extraction and classification
        4. Python for deterministic score calculation
        """

        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )

        self.retriever = ResumeRetriever()

        with open(
            jd_metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.jd_metadata = json.load(
                file
            )

    # -----------------------------------------------------
    # JSON EXTRACTION
    # -----------------------------------------------------

    def extract_json(self, text):
        """
        Extract JSON from Gemini's response.

        Handles both:
        1. Plain JSON
        2. JSON wrapped inside markdown code fences
        """

        text = text.strip()

        # Remove markdown code fences if Gemini
        # returns something like:
        #
        # ```json
        # [...]
        # ```

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

        try:
            return json.loads(text)

        except json.JSONDecodeError:

            # Try to locate the first JSON array.
            start = text.find("[")

            end = text.rfind("]")

            if start != -1 and end != -1:
                candidate = text[
                    start:end + 1
                ]

                return json.loads(
                    candidate
                )

            # Try to locate a JSON object.
            start = text.find("{")

            end = text.rfind("}")

            if start != -1 and end != -1:
                candidate = text[
                    start:end + 1
                ]

                return json.loads(
                    candidate
                )

            raise ValueError(
                "Could not extract valid JSON "
                "from Gemini response."
            )

    # -----------------------------------------------------
    # JD CONTEXT
    # -----------------------------------------------------

    def get_jd_context(self, jd_text=None):
        """
        Get the job description context.

        If the user supplies a JD directly, use that text.

        Otherwise use the JD chunks created during
        jd_ingest.py.
        """

        if jd_text and jd_text.strip():

            return jd_text.strip()

        sections = []

        for item in self.jd_metadata:

            section = item.get(
                "section",
                "GENERAL"
            )

            text = item.get(
                "text",
                ""
            )

            if text:
                sections.append(
                    f"[{section}]\n{text}"
                )

        return "\n\n".join(
            sections
        )

    # -----------------------------------------------------
    # REQUIREMENT EXTRACTION
    # -----------------------------------------------------

    def extract_requirements(
        self,
        jd_text=None
    ):
        """
        Extract important requirements from the JD.

        Gemini identifies requirements, while the actual
        match score is calculated separately in Python.
        """

        jd_context = self.get_jd_context(
            jd_text
        )

        prompt = f"""
You are analyzing a software engineering job description.

Extract the important candidate requirements from
the job description below.

Focus on requirements that can actually be evaluated
against a candidate's resume, such as:

- education
- programming languages
- technical skills
- software development
- testing and debugging
- cloud technologies
- databases
- development practices
- problem solving
- projects
- communication
- collaboration
- AI-assisted development
- deployment
- other explicitly relevant qualifications

Do not include:
- company marketing
- generic company descriptions
- benefits
- office information
- salary
- unrelated information

Keep each requirement short and specific.

Assign each requirement one importance value:

"high"
"medium"
"low"

Return ONLY valid JSON.

Required format:

[
  {{
    "id": 1,
    "requirement": "Short requirement",
    "importance": "high"
  }}
]

Job description:

{jd_context}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={"thinking_level": "low"},
        )
        output_text = interaction.output_text

        requirements = self.extract_json(
            output_text
        )

        if not isinstance(
            requirements,
            list
        ):
            raise ValueError(
                "Requirement extraction did not "
                "return a JSON list."
            )

        cleaned_requirements = []

        for index, item in enumerate(
            requirements,
            start=1
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            requirement = str(
                item.get(
                    "requirement",
                    ""
                )
            ).strip()

            importance = str(
                item.get(
                    "importance",
                    "medium"
                )
            ).lower().strip()

            if not requirement:
                continue

            if importance not in {
                "high",
                "medium",
                "low"
            }:
                importance = "medium"

            cleaned_requirements.append(
                {
                    "id": index,
                    "requirement": requirement,
                    "importance": importance
                }
            )

        return cleaned_requirements

    # -----------------------------------------------------
    # RESUME EVIDENCE RETRIEVAL
    # -----------------------------------------------------

    def retrieve_evidence(
        self,
        requirements,
        top_k=3
    ):
        """
        Retrieve resume evidence for every JD requirement.

        This is the RAG step:

        JD requirement
              ↓
        embedding
              ↓
        FAISS
              ↓
        relevant resume chunks
        """

        evidence = []

        for requirement in requirements:

            query = requirement[
                "requirement"
            ]

            results = self.retriever.search(
                query,
                top_k=top_k
            )

            evidence.append(
                {
                    "id": requirement["id"],
                    "requirement": requirement[
                        "requirement"
                    ],
                    "importance": requirement[
                        "importance"
                    ],
                    "resume_evidence": results
                }
            )

        return evidence

    # -----------------------------------------------------
    # REQUIREMENT CLASSIFICATION
    # -----------------------------------------------------

    def classify_requirements(
        self,
        evidence
    ):
        """
        Classify all requirements in one Gemini call.

        Possible statuses:

        strong
        partial
        missing
        """

        requirement_blocks = []

        for item in evidence:

            evidence_text = []

            for result in item[
                "resume_evidence"
            ]:

                evidence_text.append(
                    (
                        f"Chunk ID: "
                        f"{result['chunk_id']}\n"
                        f"Section: "
                        f"{result['section']}\n"
                        f"Similarity: "
                        f"{result['score']:.4f}\n"
                        f"Text:\n"
                        f"{result['text']}"
                    )
                )

            if not evidence_text:

                evidence_text.append(
                    "No relevant resume evidence found."
                )

            block = (
                f"Requirement ID: {item['id']}\n"
                f"Requirement: {item['requirement']}\n"
                f"Importance: {item['importance']}\n"
                f"Resume Evidence:\n"
                + "\n\n".join(evidence_text)
            )

            requirement_blocks.append(
                block
            )

        all_requirements = "\n\n".join(
            requirement_blocks
        )

        prompt = f"""
You are evaluating a candidate's resume against
software engineering job requirements.

For every requirement, classify the candidate as:

"strong"
    Clear and direct evidence exists in the resume.

"partial"
    Some relevant evidence exists, but the requirement
    is only partly demonstrated.

"missing"
    The resume does not provide sufficient evidence
    for the requirement.

Important rules:

1. Use ONLY the provided resume evidence.
2. Do not invent candidate experience.
3. Do not assume that knowing one technology means
   the candidate knows another.
4. Do not treat weak semantic similarity as proof.
5. If evidence is related but incomplete, use "partial".
6. If there is no meaningful evidence, use "missing".
7. Return exactly one result for every requirement.

Return ONLY valid JSON.

Required format:

[
  {{
    "id": 1,
    "status": "strong",
    "reason": "Short explanation based on the evidence"
  }}
]

Requirements and retrieved resume evidence:

{all_requirements}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={"thinking_level": "low"},
        )
        output_text = interaction.output_text

        classifications = self.extract_json(
            output_text
        )

        if not isinstance(
            classifications,
            list
        ):
            raise ValueError(
                "Requirement classification did not "
                "return a JSON list."
            )

        classification_map = {}

        for item in classifications:

            if not isinstance(
                item,
                dict
            ):
                continue

            try:
                item_id = int(
                    item.get("id")
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            status = str(
                item.get(
                    "status",
                    "missing"
                )
            ).lower().strip()

            if status not in {
                "strong",
                "partial",
                "missing"
            }:
                status = "missing"

            reason = str(
                item.get(
                    "reason",
                    ""
                )
            ).strip()

            classification_map[
                item_id
            ] = {
                "status": status,
                "reason": reason
            }

        final_results = []

        for item in evidence:

            classification = (
                classification_map.get(
                    item["id"],
                    {
                        "status": "missing",
                        "reason": (
                            "No classification "
                            "was returned."
                        )
                    }
                )
            )

            final_results.append(
                {
                    "id": item["id"],
                    "requirement": item[
                        "requirement"
                    ],
                    "importance": item[
                        "importance"
                    ],
                    "status": classification[
                        "status"
                    ],
                    "reason": classification[
                        "reason"
                    ],
                    "resume_evidence": item[
                        "resume_evidence"
                    ]
                }
            )

        return final_results

    # -----------------------------------------------------
    # SCORE CALCULATION
    # -----------------------------------------------------

    def calculate_score(
        self,
        results
    ):
        """
        Calculate a weighted match score.

        Importance weights:

        high   = 3
        medium = 2
        low    = 1

        Status scores:

        strong  = 1.0
        partial = 0.5
        missing = 0.0
        """

        importance_weights = {
            "high": 3,
            "medium": 2,
            "low": 1
        }

        status_scores = {
            "strong": 1.0,
            "partial": 0.5,
            "missing": 0.0
        }

        total_weight = 0.0
        achieved_weight = 0.0

        for item in results:

            importance = item[
                "importance"
            ]

            status = item[
                "status"
            ]

            weight = importance_weights.get(
                importance,
                1
            )

            status_score = status_scores.get(
                status,
                0.0
            )

            total_weight += weight

            achieved_weight += (
                weight * status_score
            )

        if total_weight == 0:

            return 0.0

        score = (
            achieved_weight
            / total_weight
        ) * 100

        return round(
            score,
            1
        )

    # -----------------------------------------------------
    # COMPLETE ANALYSIS
    # -----------------------------------------------------

    def analyze(
        self,
        jd_text=None
    ):
        """
        Run the complete resume-JD matching pipeline.

        Pipeline:

        JD
        ↓
        Requirement extraction
        ↓
        Resume retrieval
        ↓
        Evidence collection
        ↓
        Gemini classification
        ↓
        Weighted score
        """

        requirements = (
            self.extract_requirements(
                jd_text
            )
        )

        evidence = (
            self.retrieve_evidence(
                requirements,
                top_k=3
            )
        )

        results = (
            self.classify_requirements(
                evidence
            )
        )

        score = (
            self.calculate_score(
                results
            )
        )

        strong = [
            item
            for item in results
            if item["status"] == "strong"
        ]

        partial = [
            item
            for item in results
            if item["status"] == "partial"
        ]

        missing = [
            item
            for item in results
            if item["status"] == "missing"
        ]

        return {
            "score": score,
            "requirements": results,
            "strong": strong,
            "partial": partial,
            "missing": missing
        }