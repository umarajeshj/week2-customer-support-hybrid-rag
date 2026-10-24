# SupportSense AI — ShopEasy Customer Support

SupportSense AI is a beginner-friendly customer-support chatbot built with
Streamlit and retrieval-augmented generation (RAG). It searches ShopEasy's
knowledge base, then uses an OpenAI chat model to draft an answer grounded in
the retrieved information. The app also includes a knowledge-base browser and
a 20-question evaluation page.

## How the workflow works

1. **Prepare the knowledge base.** The supplied
   `data/shopeasy_knowledge_base.json` contains support articles and metadata.
2. **Create a Pinecone vector index.** The index stores semantic embeddings of
   the knowledge-base text.
3. **Ingest the articles.** The app splits articles into smaller chunks,
   creates embeddings with OpenAI, and stores the chunks in Pinecone.
4. **Ask a question.** The app retrieves relevant chunks in two ways:
   - **Vector search** finds text with similar meaning in Pinecone.
   - **BM25 search** finds keyword matches from the local JSON file.
5. **Generate and inspect an answer.** LangChain combines the search results,
   and an OpenAI chat model produces an answer. The app displays the retrieved
   evidence alongside the conversation.
6. **Evaluate the pipeline.** The Evaluation page runs 20 predefined support
   questions and reports which answers meet the expected document and phrase
   checks.

## What you need

- Windows, macOS, or Linux with Python 3.10 or newer
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone API key](https://app.pinecone.io/)
- Internet access for package installation and the OpenAI and Pinecone services

API usage may incur charges. Keep your API keys private; do not commit them or
share them in screenshots.

## Set up

Open a terminal in the project folder and create a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, allow it just for the current
terminal session, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a file named `.env` in the project root and add your API keys:

```dotenv
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

The app uses the Pinecone index `shopeasy-basic-rag` by default. You can
optionally set `PINECONE_INDEX_NAME` in `.env` if you already have a matching
Pinecone index. The index must use **512 dimensions** to match the embedding
configuration.

## Create the Pinecone index

For a first-time setup, create the default index once:

```powershell
python init_index.py
```

The script creates a 512-dimensional cosine index named `shopeasy-basic-rag`
in AWS `us-east-1`. If that index already exists, do not run the script again;
use the existing index.

## Start the app

From the project root, with the virtual environment active, run:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` helps ensure Streamlit runs with the Python from the
active virtual environment. Streamlit prints a local URL in the terminal; open
that URL in your browser. The left sidebar contains **Chat**, **Knowledge
Base**, and **Evaluation**.

## First-time knowledge-base setup

The app can search Pinecone for vector results, but the index needs to contain
the knowledge-base vectors first.

1. Open **Knowledge Base** from the sidebar.
2. In **Upload & Ingest to Pinecone**, upload
   `data/shopeasy_knowledge_base.json`.
3. Click **Ingest Documents into Pinecone** and wait for the success message.

Ingestion creates OpenAI embeddings and writes the resulting chunks to the
configured Pinecone index. The local BM25 keyword search reads
`data/shopeasy_knowledge_base.json`, so keep that file in place for hybrid
search. The **Browse Knowledge Base** tab lets you search and filter its
articles.

## Use the chatbot

1. Select **Chat** in the sidebar.
2. Enter a ShopEasy support question in the chat box and submit it.
3. Read the response and its source article titles.
4. Review **Retrieval evidence** to see the matching knowledge-base content
   and, when available, compare the vector and BM25 results.

The chat requires valid OpenAI and Pinecone keys and a populated Pinecone index.
The app's prompt asks the model to acknowledge when the retrieved context does
not contain enough information.

## Run the evaluation

Select **Evaluation (20 Queries)** in the sidebar and click **Run 20-Query
Evaluation Suite**. The evaluation compares each response against the expected
answer/escalation, expected document title, and any required phrases defined
in `eval_set.py`. It calls the live retrieval and chat services, so it also
requires working API keys and an initialized Pinecone index.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, retrieval setup, chat, knowledge-base tools, and evaluation page |
| `data/shopeasy_knowledge_base.json` | Local source data for knowledge-base browsing and BM25 keyword search |
| `init_index.py` | Creates the default Pinecone vector index |
| `ingestion_service.py` | Loads JSON articles, chunks them, creates embeddings, and uploads them to Pinecone |
| `eval_set.py` | The 20 evaluation questions and expected outcomes |
| `eval.py` | Runs the evaluation cases against the retrieval and answer-generation pipeline |
| `requirements.txt` | Python package dependencies |
| `.env` | Your local API keys; keep this file private and out of version control |

## Troubleshooting

- **Streamlit launcher says it cannot find Python:** Activate the project's
  `.venv` and run `python -m streamlit run app.py`. If the project folder or
  virtual environment was moved, recreate `.venv` in the current project
  folder, then reinstall requirements.
- **The app reports missing API keys or the RAG pipeline is unavailable:**
  Check that `.env` is in the project root, that both keys are set correctly,
  and that you restarted Streamlit after changing `.env`.
- **Pinecone reports that an index does not exist:** Run `python init_index.py`
  once, or check that `PINECONE_INDEX_NAME` matches an existing index.
- **Chat returns no useful evidence:** Confirm ingestion completed
  successfully, the app is using the same Pinecone index that was ingested,
  and `data/shopeasy_knowledge_base.json` is present for BM25 retrieval.
- **PowerShell does not recognize `python`:** Try `py` for Python commands and
  confirm that Python is installed and available on your PATH.
