# MediEasy

MediEasy is a retrieval-augmented medical question-answering application. It loads medical PDFs, splits their text into overlapping chunks, creates local vector embeddings with Sentence Transformers, stores them in FAISS, and uses a Hugging Face-hosted language model to answer questions from the retrieved context.

The Streamlit interface displays the answer and the source chunks used for retrieval.

## Features

- PDF ingestion with PyMuPDF
- Recursive text chunking with overlap
- Local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- FAISS similarity search
- Hugging Face Qwen inference through the Featherless provider
- Context-only answering with an `I don't know.` fallback
- Retrieved source files, pages, and text chunks shown in the UI
- Streamlit chat interface with session history

## Project Structure

```text
MediEasy/
|-- Data/                         # Source PDF files
|-- vectorstore/db_faiss/        # Generated FAISS index (ignored by Git)
|-- create_memory_for_llm.py      # Build the document index
|-- MedibotUI.py                  # Run the Streamlit application
|-- connect_memory_with_llm.py    # Command-line retrieval test
|-- requirements.txt              # Python dependencies
|-- Pipfile                       # Pipenv dependency declarations
|-- .env                          # Local secrets; do not commit
|-- .gitignore                    # Local and generated files excluded from Git
```

## Requirements

- Windows, macOS, or Linux
- Python 3.13
- A Hugging Face account and an access token with Inference Providers permission
- Internet access for downloading the embedding model and calling the hosted LLM
- Enough disk space for the embedding model and generated FAISS index

## Installation

From the project directory:

```powershell
cd D:\MediEasy
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project also contains a `Pipfile`. If Pipenv is installed, this is an alternative:

```powershell
pipenv install
```

If `pipenv` is not on your `PATH`, use the virtual-environment commands above.

## Configure Hugging Face

Create a `.env` file in the project root:

```dotenv
HF_TOKEN=hf_your_token_here
```

The token must be permitted to call Hugging Face Inference Providers. Never commit `.env`, paste a token into source code, or share it in screenshots. If a token is exposed, revoke it and create a replacement immediately.

## Build The Vector Store

Place one or more `.pdf` files in `Data/`, then run:

```powershell
.\.venv\Scripts\python.exe create_memory_for_llm.py
```

This creates or replaces the FAISS files under `vectorstore/db_faiss/`. Re-run this command whenever the source PDFs change.

The source files in `Data/` and the generated FAISS index are excluded by `.gitignore`. Each local checkout must provide its own PDFs and rebuild the index.

The current index uses:

- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

## Run MediBot

Start the Streamlit application with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run MedibotUI.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

Ask a question covered by the PDFs. Expand the **Sources** section below the answer to inspect the retrieved files, page numbers, and chunks.

## Command-Line Test

After building the vector store and configuring `HF_TOKEN`, run:

```powershell
.\.venv\Scripts\python.exe connect_memory_with_llm.py
```

Enter a question when prompted.

## Git And Local Files

The repository ignores files that are local, sensitive, or generated during development:

- `.env` and Streamlit secrets
- Virtual environments and Python cache directories
- Test coverage, build, packaging, and log files
- The generated `vectorstore/db_faiss/` index
- PDF, CSV, and DOCX files placed in `Data/`

Keep credentials and private medical documents out of version control. The included `.env.example` can be used as a safe template for required environment variables.

## Answering Behavior

MediBot is instructed to answer only from retrieved context. For questions that are unrelated to the indexed medical documents, or whose answer is not directly supported by the retrieved chunks, it should return:

```text
I don't know.
```

This behavior depends on retrieval quality and model compliance. The prompt is not a substitute for a hard security or clinical guarantee.

## Troubleshooting

### `HF_TOKEN` is missing or authentication fails

Check the `.env` file and restart Streamlit after changing it:

```powershell
if ($env:HF_TOKEN) { "HF_TOKEN is set" } else { "HF_TOKEN is missing" }
```

The app loads `.env` through `python-dotenv`. A token in `.env` is available to the app even when it is not set in the PowerShell session.

### Vector store not found

Build it first:

```powershell
.\.venv\Scripts\python.exe create_memory_for_llm.py
```

Confirm that `vectorstore/db_faiss/` exists.

### `torchvision` or Transformers import errors

Reinstall the declared dependencies inside the active `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### The answer is unrelated or unsupported

Confirm that the relevant information exists in a PDF under `Data/`, rebuild the vector store, and inspect the displayed source chunks. The application is a document-grounded prototype, not a medical diagnosis or emergency service.

## Safety Notice

MediEasy is an educational software project and must not replace a qualified healthcare professional. Do not use it for diagnosis, treatment decisions, medication changes, or emergencies. Always verify medical information with a licensed clinician and use local emergency services when appropriate.

## AUTHOR
ROKHIT ROY
