# HR Policy Question Answering System (RAG)

A small Streamlit application that retrieves passages from a persisted Chroma database and asks a local Hugging Face causal language model to answer an HR-policy question. The repository also contains a Kaggle-oriented notebook for rebuilding a vector store from policy PDFs.

## What is included

- `app.py` — Streamlit UI, retrieval, and answer generation.
- `chroma_hr_db/` — persisted Chroma database loaded by the app.
- `hr-policy-question-answering-system-rag.ipynb` — PDF ingestion, chunking, embedding, vector-store creation, and example queries.
- `requirements.txt` — bounded dependencies for the Streamlit app. It is not a complete lockfile; PyTorch installation varies by platform and accelerator.

The app uses `sentence-transformers/all-MiniLM-L6-v2` for query embeddings and `microsoft/Phi-3-mini-4k-instruct` for generation. The notebook instead uses a Llama 3 8B quantized model. These are two separate paths, not a multi-agent system.

## Run the app

Use Python 3.10 or newer. Install a PyTorch build appropriate for your operating system and accelerator (see the [official PyTorch install selector](https://pytorch.org/get-started/locally/)); then install the remaining requirements and launch Streamlit:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The first question may download the embedding and generation models from Hugging Face and requires network access. The generation model is several billion parameters; a compatible GPU is recommended, and CPU execution may require substantial memory and be slow. `device_map="auto"` lets Transformers use available hardware, but does not guarantee the model will fit on a given machine.

The app expects the checked-in `chroma_hr_db/` directory next to `app.py`. It displays the source filename and page metadata available from retrieved records; the source is reduced to a filename rather than displaying an absolute path.

## Rebuild the vector store from PDFs

The notebook expects policy PDFs in a directory. The PDFs are not included in this repository, and the original Kaggle input path is not available on other machines. Set `HR_POLICY_PDF_DIR` to a directory containing the authorized PDF files before running the data-loading cell, for example:

```bash
export HR_POLICY_PDF_DIR="$PWD/policy-pdfs"
```

On Kaggle, the default remains `/kaggle/input/hr-policy-docs-pdf`. The notebook reports a clear error if the selected directory is missing or contains no PDFs. Do not commit confidential or personally identifiable policy documents without authorization. Rebuilding requires access to the source PDFs and the notebook's separate model/dependency setup.

## Limitations and responsible use

- This is a basic retrieval-augmented question-answering demo, not a LangGraph workflow, multi-agent router, or IT assistant.
- A prompt asking the model to use retrieved context is not a security boundary and does not guarantee factuality or prevent prompt injection. Verify important answers against the source policy.
- The app has not been benchmarked for answer quality, retrieval accuracy, latency, or production readiness. The notebook's example queries are demonstrations, not automated performance tests.
- The app and notebook use different generation models and may produce different results. No model-performance claim is made here.
- HR policies may contain sensitive information. Run only with documents and environments approved by your organization; avoid sharing questions, outputs, or policy text with unauthorized parties.

## Local checks

The repository includes lightweight unit tests for answer formatting and source-label privacy that do not load models or require the large ML dependencies:

```bash
python -m unittest discover -s tests
python -m py_compile app.py rag_utils.py
```

## Author

Abdul Rahman Ahmed
