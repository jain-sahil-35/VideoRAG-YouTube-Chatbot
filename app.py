import re
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# ---------------- SETUP ----------------
load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 YouTube Video Q&A")
st.caption("Ask questions based **only** on the video's transcript")


# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None


# ---------------- HELPERS ----------------
def extract_video_id(url_or_id: str):
    patterns = [
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)

    except VideoUnavailable:
        return None, "Video unavailable."

    except TranscriptsDisabled:
        return None, "Captions disabled."

    except NoTranscriptFound:
        return None, "No English transcript found."

    except Exception as e:
        return None, f"Error: {str(e)}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, None


with st.sidebar:
    st.header("Video Input")

    video_url = st.text_input(
        "YouTube Video Link",
        placeholder="https://www.youtube.com/watch?v=xxxx"
    )

    load_video = st.button("Load Video")

    if load_video:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            with st.spinner("Fetching transcript..."):
                vector_store, error = build_vector_store(video_id)

                if error:
                    st.error(error)
                else:
                    st.success("Transcript loaded successfully")
                    st.session_state.vector_store = vector_store
                    st.session_state.current_video_id = video_id
                    st.session_state.chat_history.clear()


    st.markdown("---")
    st.caption("Answers are generated **only from transcript context**")


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.chat_input("Ask something about the video...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please load a video first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # RAG PIPELINE
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, answer in a well structured manner.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-safeguard-20b",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }) | prompt | model | parser
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            st.markdown(answer)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })
