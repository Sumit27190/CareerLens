import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import faiss
import pymupdf
import streamlit as st

from rag.matcher import ResumeMatcher
from rag.interviewer import Interviewer
from rag.retrieval import ResumeRetriever
from ingest import create_chunks, extract_blocks


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Career Lens",
    page_icon="CL",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17202a;
        --muted: #667481;
        --line: #dce3e8;
        --paper: #ffffff;
        --accent: #d68a38;
        --accent-soft: #fff3e5;
        --success: #197a55;
        --warning: #a96818;
        --danger: #b74a4a;
    }

    /* ---------- Page ---------- */

    .stApp {
        background: #f3f5f7;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--ink) !important;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 34px;
        padding-bottom: 70px;
    }

    p, label, .stCaption, [data-testid="stMarkdownContainer"] {
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: #17202a;
        border-right: 1px solid #2b3b48;
    }

    [data-testid="stSidebar"] * {
        color: #e7edf2;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: #30414e;
        margin: 1.25rem 0;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }

    .sidebar-mark {
        display: grid;
        place-items: center;
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: #e3a857;
        color: #17202a;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 15px;
        font-weight: 700;
    }

    .sidebar-brand-name {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
    }

    .sidebar-copy {
        color: #94a5b0;
        font-size: 12px;
        line-height: 1.55;
        margin: 4px 0 10px 44px;
    }

    .workflow-title {
        color: #e3a857;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .workflow-step {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #d7e0e5;
        font-size: 13px;
        padding: 5px 0;
    }

    .workflow-number {
        display: grid;
        place-items: center;
        width: 23px;
        height: 23px;
        border: 1px solid #526674;
        border-radius: 50%;
        color: #e3a857;
        font-size: 11px;
        font-weight: 700;
    }

    .sidebar-footer {
        color: #70818d;
        font-size: 11px;
        line-height: 1.5;
        padding-top: 4px;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #263746 !important;
        border-color: #415467 !important;
        color: #ffffff !important;
    }

    /* ---------- Hide Streamlit chrome ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ---------- Hero ---------- */

    .hero-container {
        background: linear-gradient(120deg, #17202a 0%, #294452 72%, #3c5860 100%);
        border-radius: 8px;
        padding: 38px 42px;
        margin-bottom: 34px;
        border-left: 6px solid #e3a857;
        box-shadow: 0 14px 32px rgba(23, 32, 42, 0.12);
    }

    .hero-container h1 {
        color: white !important;
        font-size: 34px !important;
        font-weight: 750 !important;
        margin: 8px 0 8px 0 !important;
        letter-spacing: 0;
    }

    .hero-container p {
        color: #c2cdd1 !important;
        font-size: 15px !important;
        max-width: 720px;
        line-height: 1.6;
    }

    .hero-badge {
        color: #cbd5e1;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.8px;
    }

    /* ---------- Section headings ---------- */

    h2 {
        font-size: 25px !important;
        margin-top: 34px !important;
        margin-bottom: 12px !important;
    }

    h3 {
        font-size: 18px !important;
    }

    /* ---------- Text area ---------- */

    textarea {
        background-color: var(--paper) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        font-size: 14px !important;
        line-height: 1.55 !important;
        box-shadow: 0 3px 12px rgba(23, 32, 42, 0.04);
    }

    textarea:focus {
        border-color: #64748b !important;
        box-shadow: 0 0 0 1px #64748b !important;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        background-color: var(--ink) !important;
        color: white !important;
        border: 1px solid var(--ink) !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
        transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
    }

    .stButton > button:hover {
        background-color: #294452 !important;
        border-color: #294452 !important;
        box-shadow: 0 5px 14px rgba(23, 32, 42, 0.18);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        background: var(--paper);
        border: 1px dashed #b8c5cc;
        border-radius: 8px;
        padding: 10px 12px 4px;
        box-shadow: 0 3px 12px rgba(23, 32, 42, 0.04);
    }

    div[data-testid="stFileUploader"] section {
        padding: 4px 0;
    }

    /* ---------- Metric boxes ---------- */

    .metric-box {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 8px;
        padding: 22px;
        min-height: 110px;
        box-shadow: 0 4px 16px rgba(23, 32, 42, 0.05);
    }

    .metric-title {
        color: #667085;
        font-size: 12px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 32px;
        font-weight: 800;
        margin-top: 7px;
    }

    .metric-green {
        color: #16a34a;
    }

    .metric-yellow {
        color: #d97706;
    }

    .metric-red {
        color: #dc2626;
    }

    /* ---------- Score box ---------- */

    .score-box {
        background: #17202a;
        border-radius: 8px;
        padding: 22px;
        min-height: 110px;
        box-shadow: 0 7px 20px rgba(23, 32, 42, 0.13);
    }

    .score-title {
        color: #9ca3af;
        font-size: 12px;
        font-weight: 600;
    }

    .score-value {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-top: 7px;
    }

    /* ---------- Requirement boxes ---------- */

    .requirement-box {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 8px;
        padding: 17px;
        margin-bottom: 10px;
        box-shadow: 0 3px 12px rgba(23, 32, 42, 0.04);
    }

    .requirement-title {
        color: #1f2937;
        font-size: 14px;
        font-weight: 650;
        line-height: 1.5;
    }

    .requirement-evidence {
        color: #667085;
        font-size: 12px;
        line-height: 1.5;
        margin-top: 8px;
    }

    /* ---------- Interview question ---------- */

    .interview-box {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 8px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 4px 16px rgba(23, 32, 42, 0.05);
    }

    .interview-label {
        color: #98a2b3;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .interview-question {
        color: #111827;
        font-size: 19px;
        font-weight: 700;
        line-height: 1.55;
        margin-top: 8px;
    }

    /* ---------- Answer ---------- */

    .answer-box {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 8px;
        padding: 22px;
        margin-top: 15px;
        box-shadow: 0 4px 16px rgba(23, 32, 42, 0.05);
    }

    .answer-title {
        color: #111827;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 10px;
    }

    .answer-text {
        color: #374151;
        font-size: 14px;
        line-height: 1.75;
    }

    /* ---------- Preparation ---------- */

    .prep-box {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 10px;
        box-shadow: 0 3px 12px rgba(23, 32, 42, 0.04);
    }

    .prep-label {
        color: #98a2b3;
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 1px;
    }

    .prep-title {
        color: #111827;
        font-size: 15px;
        font-weight: 700;
        margin-top: 5px;
    }

    .prep-description {
        color: #667085;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 7px;
    }

    /* ---------- Selectbox ---------- */

    div[data-baseweb="select"] > div {
        background-color: var(--paper) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: var(--muted) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 6px;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #9ca3af !important;
    }

    .eyebrow {
        color: #e3a857;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .stProgress > div > div > div > div {
        background-color: #e3a857;
    }

    @media (max-width: 700px) {
        .block-container { padding: 18px 18px 45px; }
        .hero-container { padding: 26px 24px; }
        .hero-container h1 { font-size: 28px !important; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD BACKEND
# ============================================================

@st.cache_resource
def get_matcher():
    loaded_matcher = ResumeMatcher()
    loaded_matcher.default_retriever = loaded_matcher.retriever
    return loaded_matcher


@st.cache_resource
def get_interviewer():
    return Interviewer()


interviewer = None


# ============================================================
# SESSION STATE
# ============================================================

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answer" not in st.session_state:
    st.session_state.answer = None

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

def get_cached_interviewer():
    global interviewer
    if interviewer is None:
        interviewer = get_interviewer()
    interviewer.retriever = matcher.retriever
    return interviewer


def quiet_call(function, *args, **kwargs):
    output = StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        return function(*args, **kwargs)


def explain_api_error(error):
    message = str(error)
    if "403" in message or "permission_denied" in message:
        return (
            "Gemini denied access to this project or model. "
            "Check that the API key belongs to the correct Google Cloud "
            "project and that the selected model is enabled."
        )
    if "429" in message or "quota" in message.lower():
        return (
            "Gemini API quota is temporarily exhausted. "
            "Wait for the retry period shown by Google, then try again. "
            "You can also check your usage and billing at ai.google.dev."
        )
    return f"{error}"


def read_uploaded_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in document)
        document.close()
        return text.strip()
    return file_bytes.decode("utf-8", errors="replace").strip()


def build_uploaded_retriever(uploaded_file, model):
    temp_dir = tempfile.mkdtemp(prefix="career_lens_")
    resume_path = os.path.join(temp_dir, uploaded_file.name)
    index_path = os.path.join(temp_dir, "resume.index")
    metadata_path = os.path.join(temp_dir, "metadata.json")

    with open(resume_path, "wb") as file:
        file.write(uploaded_file.getvalue())

    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Resume upload must be a PDF so section formatting can be read.")

    chunks = create_chunks(extract_blocks(resume_path))
    if not chunks:
        raise ValueError("No readable resume text was found in that PDF.")

    embeddings = quiet_call(
        model.encode,
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)

    metadata = []
    for chunk_id, chunk in enumerate(chunks):
        item = {
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "section": chunk["section"],
            "source": uploaded_file.name,
        }
        if "project" in chunk:
            item["project"] = chunk["project"]
        metadata.append(item)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file)

    return ResumeRetriever(index_path=index_path, metadata_path=metadata_path)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="eyebrow">
            Evidence-led career preparation
        </div>
        <h1>See how your experience fits the role.</h1>
        <p>Turn a job description into a practical match report, interview questions, and a focused preparation plan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-mark">CL</div>
            <div class="sidebar-brand-name">Career Lens</div>
        </div>
        <div class="sidebar-copy">
            Turn your resume and a target role into a focused preparation plan.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Start new session", use_container_width=True):
        for state_key in (
            "analysis",
            "questions",
            "answer",
            "custom_answer",
            "selected_question",
            "custom_question",
            "custom_answer",
            "custom_question_answered",
            "jd_editor",
            "resume_upload",
            "jd_upload",
        ):
            st.session_state.pop(state_key, None)
        st.rerun()
    st.divider()
    st.markdown('<div class="workflow-title">Workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-step"><span class="workflow-number">1</span><span>Add your files</span></div>
        <div class="workflow-step"><span class="workflow-number">2</span><span>Review the evidence</span></div>
        <div class="workflow-step"><span class="workflow-number">3</span><span>Practice questions</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        '<div class="sidebar-footer">Powered by resume retrieval and Gemini<br>Session data stays in this browser.</div>',
        unsafe_allow_html=True,
    )

st.header("Your files")
file_col1, file_col2 = st.columns(2)
with file_col1:
    uploaded_resume = st.file_uploader(
        "Upload resume",
        type=["pdf"],
        key="resume_upload",
        help="Upload a PDF resume to use instead of the indexed resume.",
    )
with file_col2:
    uploaded_jd = st.file_uploader(
        "Upload job description",
        type=["pdf", "txt"],
        key="jd_upload",
        help="Upload a PDF or TXT job description, or paste it below.",
    )


# Load the embedding model after the shell is rendered so startup has a
# useful visible state instead of showing a blank page.
with st.status("Preparing your workspace...", expanded=False) as startup_status:
    try:
        matcher = get_matcher()
        if uploaded_resume is not None:
            with st.spinner("Indexing uploaded resume..."):
                matcher.retriever = build_uploaded_retriever(
                    uploaded_resume,
                    matcher.retriever.model,
                )
        else:
            matcher.retriever = matcher.default_retriever
        startup_status.update(
            label="Workspace ready",
            state="complete",
            expanded=False,
        )
    except Exception as error:
        startup_status.update(
            label="Resume index could not be loaded",
            state="error",
        )
        st.error(f"Startup failed: {error}")
        st.stop()


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("Job Description")

st.caption(
    "Paste the job description you want to analyze against your resume."
)

jd_text = st.text_area(
    "Job description",
    value="",
    key="jd_editor",
    height=230,
    label_visibility="collapsed",
    placeholder="Paste a job description here...",
)

if uploaded_jd is not None:
    uploaded_jd_text = read_uploaded_text(uploaded_jd)
    if uploaded_jd_text:
        jd_text = uploaded_jd_text
        st.info(f"Using uploaded job description: {uploaded_jd.name}")
    else:
        st.warning("The uploaded job description did not contain readable text.")

st.caption(f"{len(jd_text.split()):,} words loaded. You can replace the sample role description at any time.")


if st.button(
    "Analyze Resume →",
    type="primary",
    use_container_width=True,
):

    if not jd_text.strip():

        st.warning("Please enter a job description.")

    else:

        with st.spinner("Comparing your resume with the role..."):
            try:
                st.session_state.analysis = matcher.analyze(jd_text=jd_text)
            except Exception as error:
                st.session_state.analysis_error = explain_api_error(error)
                st.error(f"Analysis could not be completed: {explain_api_error(error)}")
                st.warning("Your previous results were kept. No new analysis was saved.")
            else:
                st.session_state.analysis_error = None
                st.session_state.questions = []
                st.session_state.answer = None
                st.session_state.selected_question = None
                st.rerun()


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis:

    analysis = st.session_state.analysis

    score = analysis.get("score", 0)

    classifications = analysis.get(
        "classifications",
        analysis.get("requirements", []),
    )

    strong = [
        item
        for item in classifications
        if item.get("status", "").lower() == "strong"
    ]

    partial = [
        item
        for item in classifications
        if item.get("status", "").lower() == "partial"
    ]

    missing = [
        item
        for item in classifications
        if item.get("status", "").lower() == "missing"
    ]


    # ========================================================
    # MATCH OVERVIEW
    # ========================================================

    st.header("Match Overview")

    col1, col2, col3, col4 = st.columns(
        [1.3, 1, 1, 1]
    )

    with col1:
        with st.container(border=True):
            st.metric("Resume match", f"{score:.1f}%")

    with col2:
        with st.container(border=True):
            st.metric("Strong", len(strong))

    with col3:
        with st.container(border=True):
            st.metric("Partial", len(partial))

    with col4:
        with st.container(border=True):
            st.metric("Missing", len(missing))


    # ========================================================
    # TABS
    # ========================================================

    st.write("")

    match_tab, interview_tab, ask_tab, preparation_tab = st.tabs(
        [
            "Match Analysis",
            "Interview Prep",
            "Ask Career Lens",
            "Preparation Plan",
        ]
    )


    # ========================================================
    # MATCH ANALYSIS
    # ========================================================

    with match_tab:

        st.subheader("Requirement Breakdown")

        st.caption(
            "Each requirement is evaluated against relevant evidence "
            "retrieved from your resume."
        )


        def show_requirement(item, status):

            requirement = item.get(
                "requirement",
                "Requirement",
            )

            evidence = item.get(
                "reason",
                item.get(
                    "evidence",
                    "No additional evidence.",
                ),
            )

            with st.container(border=True):
                st.markdown(f"**{requirement}**")
                st.caption(f"{status}: {evidence}")


        if strong:

            st.markdown("#### Strong matches")

            for item in strong:
                show_requirement(
                    item,
                    "Strong",
                )


        if partial:

            st.markdown("#### Partial matches")

            for item in partial:
                show_requirement(
                    item,
                    "Partial",
                )


        if missing:

            st.markdown("#### Missing requirements")

            for item in missing:
                show_requirement(
                    item,
                    "Missing",
                )


    # ========================================================
    # INTERVIEW PREPARATION
    # ========================================================

    with interview_tab:

        st.subheader("Interview Preparation")

        st.caption(
            "Questions are generated from the job description and "
            "your resume."
        )

        if st.button(
            "Generate Interview Questions →",
            type="primary",
        ):
            try:
                with st.spinner("Generating interview questions..."):
                    st.session_state.questions = get_cached_interviewer().generate_questions(
                        jd_text=jd_text,
                        count=12,
                    )
                st.session_state.answer = None
                st.session_state.selected_question = None
            except Exception as error:
                st.error(f"Questions could not be generated: {explain_api_error(error)}")


        if st.session_state.questions:

            st.write("")

            question_labels = []

            for index, question in enumerate(
                st.session_state.questions,
                start=1,
            ):

                if isinstance(question, dict):

                    question_text = question.get(
                        "question",
                        str(question),
                    )

                else:

                    question_text = str(question)

                question_labels.append(
                    f"{index}. {question_text}"
                )


            selected = st.selectbox(
                "Select a question",
                question_labels,
            )


            selected_index = question_labels.index(
                selected
            )

            selected_data = (
                st.session_state.questions[
                    selected_index
                ]
            )


            if isinstance(selected_data, dict):

                current_question = selected_data.get(
                    "question",
                    selected,
                )

            else:

                current_question = str(
                    selected_data
                )


            with st.container(border=True):
                st.caption(f"QUESTION {selected_index + 1}")
                st.markdown(f"### {current_question}")


            st.write("")


            if st.button(
                "Generate Grounded Answer →",
                type="primary",
            ):
                try:
                    with st.spinner("Preparing answer from your resume..."):
                        st.session_state.answer = get_cached_interviewer().generate_answer(
                            current_question,
                            jd_text=jd_text,
                        )
                        st.session_state.selected_question = current_question
                except Exception as error:
                    st.error(f"Answer could not be generated: {explain_api_error(error)}")


            if (
                st.session_state.answer
                and st.session_state.selected_question
                == current_question
            ):

                answer = str(
                    st.session_state.answer
                )

                with st.container(border=True):
                    st.markdown("### Suggested Answer")
                    st.markdown(answer)

                st.caption(
                    "Generated using information retrieved from your resume."
                )


        else:

            st.info(
                "Generate interview questions to start practicing."
            )

    # ========================================================
    # CUSTOM RESUME / JD QUESTIONS
    # ========================================================

    with ask_tab:

        st.subheader("Ask Career Lens")
        st.caption(
            "Ask anything about how your resume fits the job description. "
            "Answers are grounded in both documents."
        )

        custom_question = st.text_area(
            "Your resume and JD question",
            placeholder="Example: Which projects from my resume best match this job description?",
            height=130,
            label_visibility="collapsed",
            key="custom_question",
        )

        if st.button("Analyze my question", type="primary"):
            if not custom_question.strip():
                st.warning("Type a question first.")
            else:
                try:
                    with st.spinner("Comparing your resume with the job description..."):
                        st.session_state.custom_answer = get_cached_interviewer().answer_resume_jd_question(
                            custom_question.strip(),
                            jd_text=jd_text,
                        )
                        st.session_state.custom_question_answered = custom_question.strip()
                except Exception as error:
                    st.error(f"Question could not be answered: {explain_api_error(error)}")

        if (
            st.session_state.get("custom_answer")
            and st.session_state.get("custom_question_answered") == custom_question.strip()
        ):
            with st.container(border=True):
                st.markdown("### Career Lens response")
                st.markdown(st.session_state.custom_answer)


    # ========================================================
    # PREPARATION PLAN
    # ========================================================

    with preparation_tab:

        st.subheader("Preparation Roadmap")

        st.caption(
            "Focus on requirements where your resume currently "
            "has partial or missing evidence."
        )

        focus_items = missing + partial

        if not focus_items:

            st.success(
                "No major gaps were identified. Focus on interview "
                "practice and communicating your experience clearly."
            )

        else:

            for index, item in enumerate(
                focus_items,
                start=1,
            ):

                requirement = item.get(
                    "requirement",
                    "Requirement",
                )

                status = (
                    item.get(
                        "status",
                        "partial",
                    )
                    .lower()
                )

                if status == "missing":

                    description = (
                        "This requirement is not currently supported "
                        "by your resume. Learn the concept and, if "
                        "possible, build a practical example."
                    )

                else:

                    description = (
                        "You have some evidence for this requirement. "
                        "Strengthen your understanding and prepare a "
                        "clear example explaining your contribution."
                    )

                with st.container(border=True):
                    st.caption(f"FOCUS {index}")
                    st.markdown(f"**{requirement}**")
                    st.write(description)


elif st.session_state.analysis_error:

    st.info("Analysis was not completed. Resolve the issue above and try again.")

else:

    # ========================================================
    # EMPTY STATE
    # ========================================================

    st.info(
        "Paste a job description above and click "
        "**Analyze Resume** to begin."
    )