import json
import os
import re

from dotenv import load_dotenv
from google import genai

from rag.retrieval import ResumeRetriever


class Interviewer:

    def __init__(
        self,
        jd_metadata_path="vector_store/jd_metadata.json",
        model_name=None
    ):
        """
        Initialize the interview preparation system.

        The interviewer uses:
        1. Resume retrieval through FAISS
        2. JD context from the JD vector store
        3. Gemini for question generation
        4. Gemini for resume-grounded answer generation
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

        Handles:
        1. Plain JSON
        2. JSON inside markdown code fences
        """

        text = text.strip()

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

            start = text.find("[")

            end = text.rfind("]")

            if start != -1 and end != -1:

                candidate = text[
                    start:end + 1
                ]

                return json.loads(
                    candidate
                )

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

    def get_jd_context(
        self,
        jd_text=None
    ):
        """
        Return the job description context.

        If a JD is supplied directly, use it.
        Otherwise use the ingested JD chunks.
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
    # RESUME CONTEXT
    # -----------------------------------------------------

    def get_resume_context(
        self,
        query,
        top_k=5
    ):
        """
        Retrieve the most relevant resume chunks
        for an interview question.
        """

        results = self.retriever.search(
            query,
            top_k=top_k
        )

        if not results:

            return "No relevant resume evidence found."

        context_blocks = []

        for result in results:

            context_blocks.append(
                (
                    f"Chunk ID: {result['chunk_id']}\n"
                    f"Section: {result['section']}\n"
                    f"Similarity: {result['score']:.4f}\n"
                    f"Resume text:\n"
                    f"{result['text']}"
                )
            )

        return "\n\n".join(
            context_blocks
        )

    # -----------------------------------------------------
    # QUESTION GENERATION
    # -----------------------------------------------------

    def generate_questions(
        self,
        jd_text=None,
        count=12
    ):
        """
        Generate interview questions based on:

        1. The job description
        2. The candidate's resume

        Questions are generated in one Gemini call.
        """

        jd_context = self.get_jd_context(
            jd_text
        )

        resume_chunks = (
            self.retriever.get_all_metadata()
        )

        resume_context_blocks = []

        for item in resume_chunks:

            resume_context_blocks.append(
                (
                    f"Section: "
                    f"{item['section']}\n"
                    f"{item['text']}"
                )
            )

        resume_context = "\n\n".join(
            resume_context_blocks
        )

        prompt = f"""
You are an experienced software engineering
interviewer.

Generate exactly {count} interview questions
for a candidate applying to the role described
in the job description below.

The questions should be personalized using the
candidate's resume.

Cover a realistic mixture of:

- resume/project questions
- technical questions
- software engineering fundamentals
- problem-solving questions
- questions related to technologies mentioned
  in the JD
- behavioral questions
- questions about engineering decisions
- questions about testing/debugging/deployment
  when relevant
- questions that probe the candidate's actual
  project experience

Important rules:

1. Do not invent projects or experience.
2. Prefer questions that can be answered using
   the candidate's actual resume.
3. Make questions specific enough to be useful
   in an interview.
4. Avoid asking multiple versions of the same
   question.
5. Keep each question concise.
6. The questions should be appropriate for a
   software engineering internship role.

Return ONLY valid JSON.

Required format:

[
  {{
    "id": 1,
    "question": "Question text",
    "category": "project"
  }}
]

Allowed categories:

"project"
"technical"
"fundamentals"
"problem_solving"
"behavioral"
"engineering"

Job description:

{jd_context}

Candidate resume:

{resume_context}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={"thinking_level": "low"},
        )
        output_text = interaction.output_text

        questions = self.extract_json(
            output_text
        )

        if not isinstance(
            questions,
            list
        ):
            raise ValueError(
                "Question generation did not "
                "return a JSON list."
            )

        cleaned_questions = []

        for index, item in enumerate(
            questions,
            start=1
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            question = str(
                item.get(
                    "question",
                    ""
                )
            ).strip()

            category = str(
                item.get(
                    "category",
                    "technical"
                )
            ).lower().strip()

            if not question:
                continue

            allowed_categories = {
                "project",
                "technical",
                "fundamentals",
                "problem_solving",
                "behavioral",
                "engineering"
            }

            if category not in allowed_categories:

                category = "technical"

            cleaned_questions.append(
                {
                    "id": index,
                    "question": question,
                    "category": category
                }
            )

        return cleaned_questions

    # -----------------------------------------------------
    # ANSWER GENERATION
    # -----------------------------------------------------

    def generate_answer(
        self,
        question,
        jd_text=None
    ):
        """
        Generate a resume-grounded answer
        for a selected interview question.

        The question is used as the retrieval query,
        so the RAG system retrieves the most relevant
        resume evidence before Gemini generates
        the answer.
        """

        resume_context = (
            self.get_resume_context(
                question,
                top_k=5
            )
        )

        jd_context = self.get_jd_context(
            jd_text
        )

        prompt = f"""
You are helping a candidate prepare an answer
for a software engineering interview.

Answer the interview question using ONLY
the provided resume evidence.

Important rules:

1. Do not invent experience, technologies,
   responsibilities, results, metrics, or
   achievements.
2. Do not claim the candidate did something
   unless the resume evidence supports it.
3. If the resume does not contain enough
   information to answer a specific part,
   say what information is missing instead
   of making it up.
4. Give a natural spoken interview answer.
5. Keep the answer concise enough to speak
   comfortably in an interview.
6. Use the candidate's actual projects and
   technologies when relevant.
7. Prefer a clear structure:
   situation/context → what I did →
   technical details → result/learning.
8. Do not mention "RAG", "retrieved chunks",
   "resume evidence", or this prompt in the
   answer.
9. The answer should sound like the candidate
   is speaking, not like a documentation page.

Job description:

{jd_context}

Interview question:

{question}

Relevant resume evidence:

{resume_context}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={"thinking_level": "low"},
        )
        answer = interaction.output_text.strip()

        if not answer:

            raise ValueError(
                "Gemini returned an empty answer."
            )

        return answer

    def answer_resume_jd_question(
        self,
        question,
        jd_text=None
    ):
        """Answer a custom question by comparing the full resume with the JD."""

        jd_context = self.get_jd_context(jd_text)
        resume_context = "\n\n".join(
            f"Section: {item.get('section', 'GENERAL')}\n{item.get('text', '')}"
            for item in self.retriever.get_all_metadata()
        )

        prompt = f"""
You are a resume and job-description analysis assistant.

Answer the user's question by comparing the job description
with the candidate's resume below.

Important rules:
1. Use only information present in the provided resume and job description.
2. Do not invent projects, skills, tools, responsibilities, or results.
3. When identifying project matches, name the project and explain which JD
   requirement it supports using evidence from the resume.
4. Distinguish clearly between strong matches, partial matches, and gaps.
5. If there is no supported match, say so directly.
6. Use concise headings and bullet points when they improve clarity.

User question:
{question}

Job description:
{jd_context}

Candidate resume:
{resume_context}
"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
            generation_config={"thinking_level": "low"},
        )
        answer = interaction.output_text.strip()

        if not answer:
            raise ValueError("Gemini returned an empty answer.")

        return answer