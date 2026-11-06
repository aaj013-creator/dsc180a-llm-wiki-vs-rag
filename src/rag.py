"""RAG implementation with NHTSA post-2010 vehicle recall data

Requirements:
- Groq: put `GROQ_API_KEY=...` in the `.env` file
- Ollama: run `ollama serve` and `ollama pull nomic-embed-text`
"""

# Imports
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from groq import Groq

# Load GROQ_API_KEY from the .env file
load_dotenv()

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "FLAT_RCL_POST_2010.txt"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "embeddings.npy"

# Ollama endpoint
OLLAMA = "http://localhost:11434"

# Models
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "openai/gpt-oss-20b"

COLUMNS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "MFGCAMPNO",
    "COMPNAME", "MFGNAME", "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA", "RPNO", "FMVSS",
    "DESC_DEFECT", "CONEQUENCE_DEFECT", "CORRECTIVE_ACTION", "NOTES",
    "RCL_CMPT_ID", "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE",
]


# Embedding and LLM Setup
def embed(texts):
    """Embed a list of texts using the Ollama embedding model"""
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    return np.array(r.json()["embeddings"])

def ask_llm(prompt):
    """Send a prompt to the Groq chat model and return its reply"""
    client = Groq()

    try:
        r = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")


# Data loading
def load_data():
    """Load the raw recall data into a DataFrame"""
    return pd.read_csv(
        DATA_PATH,
        sep="\t",  # Data is stored as .txt but is separated by tabs
        header=None,
        names=COLUMNS,
        dtype=str,
        encoding="latin-1",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )

def build_doc(g):
    """Build a document from a campaign's rows"""
    first = g.iloc[0]
    models = sorted(set(g["MAKETXT"] + " " + g["MODELTXT"] + " " + g["YEARTXT"]))
    text = (
        f"Recall campaign {first['CAMPNO']}.\n"
        f"Manufacturer: {first['MFGNAME']}.\n"
        f"Component: {first['COMPNAME']}.\n"
        f"Affected vehicles: {'; '.join(models)}.\n"
        f"Defect: {first['DESC_DEFECT']}\n"
        f"Consequence: {first['CONEQUENCE_DEFECT']}\n"
        f"Remedy: {first['CORRECTIVE_ACTION']}"
    )
    return {
        "id": first["CAMPNO"],
        "text": text,
        "makes": sorted(set(g["MAKETXT"])),
        "years": sorted(set(g["YEARTXT"])),
        "component": first["COMPNAME"],
    }

def build_docs(df):
    """Group the DataFrame by recall campaign and build a document for each"""
    return [build_doc(g) for _, g in df.groupby("CAMPNO")]


# Indexing
def build_index(docs, batch_size=64):
    """Embed each doc's `text`, returns (texts, vectors)

    EMBEDDINGS_PATH is used to cache the vectors re-embedding isn't required on every run,
    delete the file to force re-embedding or if data changes
    """
    texts = ["search_document: " + doc["text"] for doc in docs]

    if EMBEDDINGS_PATH.is_file():
        vectors = np.load(EMBEDDINGS_PATH)
    else:
        vectors = np.vstack(
            [embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        )
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, vectors)
        print(f"Indexed {len(docs)} chunks")

    return texts, vectors


# Retrieval & generation
def retrieve(query, texts, vectors, k=5):
    """Return the top k texts most similar via cosine similarity to the query"""
    q = embed([query])[0]
    scores = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    return [texts[i] for i in np.argsort(scores)[::-1][:k]]

def ask_llm_rag(question, texts, vectors):
    """Answer a question using the retrieved chunks as context"""
    context = "\n\n".join(retrieve(question, texts, vectors))
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context:
{context}

Question: {question}"""

    return ask_llm(prompt)


# Main
def main():
    df = load_data()
    docs = build_docs(df)
    texts, vectors = build_index(docs)

    # Example question
    question = "What is the recall for the 2015 Ford F-150?"
    print(f"Question: {question}")
    print(ask_llm_rag(question, texts, vectors))

    # Chat loop (q to quit)
    while True:
        question = input("\nAsk a question (q to quit): ")

        if question.lower() == "q":
            break

        print(ask_llm_rag(question, texts, vectors))


if __name__ == "__main__":
    main()
