import streamlit as st
import os
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


st.set_page_config(
    page_title="HR Policy Question Answering System",
    layout="wide"
)

st.title("HR Policy Assistant")

question = st.text_input("Enter your HR policy question")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_hr_db")

if not os.path.exists(CHROMA_PATH):
    st.error("Chroma database folder not found.")
    st.stop()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding = load_embeddings()

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

@st.cache_resource
def load_llm():
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.eval()

    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=generation_pipeline)

llm = load_llm()

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        documents = retriever.get_relevant_documents(question)

        if not documents:
            st.error("No relevant information was found in the HR policies.")
        else:
            context = "\n\n".join(doc.page_content for doc in documents)

            prompt = f"""
You are a professional HR assistant.
Answer the question using ONLY the information provided in the context below.

If the answer does not exist in the context, respond exactly with:
I am sorry, but this information is not available in the HR policies.

Context:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("Generating answer..."):
                response = llm(prompt)

            if isinstance(response, list):
                response = response[0]["generated_text"]

            st.subheader("Answer")
            st.write(response)

            st.subheader("Sources")
            for doc in documents:
                st.write(doc.metadata)
