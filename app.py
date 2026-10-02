import os

import streamlit as st
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from rag_utils import extract_answer, format_source_label


st.set_page_config(
    page_title="HR Policy Question Answering System",
    layout="wide",
)

st.title("HR Policy Assistant")
question = st.text_input("Enter your HR policy question").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_hr_db")

if not os.path.isdir(CHROMA_PATH):
    st.error("Chroma database folder not found.")
    st.stop()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=load_embeddings(),
    )


@st.cache_resource
def load_llm():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if torch.cuda.is_available():
        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    model.eval()

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=generation_pipeline)


if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the HR policies..."):
            vectorstore = load_vectorstore()
            documents = vectorstore.similarity_search(question, k=3)

        if not documents:
            st.error("No relevant information was found in the HR policies.")
        else:
            context = "\n\n".join(doc.page_content for doc in documents)
            prompt = f"""You are a professional HR assistant.
Answer the question using ONLY the information provided in the context below.

If the answer does not exist in the context, respond exactly with:
I am sorry, but this information is not available in the HR policies.

Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("Loading the language model and generating an answer..."):
                llm = load_llm()
                response = llm.invoke(prompt)

            answer = extract_answer(response, prompt)
            st.subheader("Answer")
            st.write(answer or "The model returned an empty answer.")

            st.subheader("Sources")
            source_labels = dict.fromkeys(
                format_source_label(doc.metadata) for doc in documents
            )
            for label in source_labels:
                st.write(label)
