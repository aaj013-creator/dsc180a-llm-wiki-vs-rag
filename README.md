# Grounding LLMs in Vehicle Safety Data: RAG vs. an LLM-Only Baseline

DSC 180A Capstone, Quarter 1 Project
Mentor: Ryan (99P Labs)
Team: Aarushi Jain, Ryan Tang, Shawn Cheng, Henry Liu

## Overview

Large language models (LLMs) can answer many general questions, but answering questions about a specific domain requires access to external knowledge. This project compares two ways of answering questions about vehicle safety recalls published by the National Highway Traffic Safety Administration (NHTSA):

1. LLM only: the model answers from its own knowledge, with no access to the recall records (baseline)
2. Naive RAG: recall records are chunked and embedded, and the most relevant chunks are retrieved and passed to the model for each question

We evaluate both systems on two tasks:

- Single-record lookup: questions answered by one recall
- Cross-record synthesis: questions that require combining information across several recalls

This baseline comparison sets up future work on more structured approaches, such as LLM-maintained knowledge bases, for reasoning over records that change over time.

## Evaluation Scorecard

| Metric | Description |
| --- | --- |
| Answer correctness | Scored by an LLM judge, spot-checked by hand |
| Groundedness | Whether the answer cites the correct recall records |
| Abstention | Whether the system says it does not know on unanswerable questions |
| Cost | Tokens used per question |

The evaluation question set is in `data/processed/questions.csv`. Each question includes a task type, a gold answer, and the NHTSA campaign numbers that contain the answer.

## Repository Structure

```
.
├── data/
│   ├── raw/          # Raw NHTSA JSON responses (not committed; download with run.py)
│   └── processed/    # Cleaned records and evaluation question set
├── notebooks/        # Exploration and analysis
├── results/          # Scorecard outputs
├── src/              # Source code (data download, baselines, evaluation)
├── run.py            # Runs the pipeline
├── requirements.txt  # Python dependencies
└── README.md
```

## Getting the Data

Data comes from the public NHTSA API (https://www.nhtsa.gov/nhtsa-datasets-and-apis). No API key is required.

Current scope:

- Make: Honda
- Models: Civic, Accord, CR-V
- Model years: 2018 to 2022
- Source used in experiments: recalls

The download script also saves owner complaints for the same vehicles, which may be used in future work.

NHTSA updates its data daily, so record counts may change slightly between downloads.

## Setup

Requires Python 3.9 or newer.

1. Clone the repository:
   ```
   git clone https://github.com/aaj013-creator/dsc180a-llm-wiki-vs-rag.git
   cd dsc180a-llm-wiki-vs-rag
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set your LLM API key as an environment variable:
   ```
   export LLM_API_KEY="your-key-here"
   ```
   Never commit API keys to the repository.

## Reproducing Results

Download the NHTSA data into `data/raw/`:

```
python run.py data
```

[TO DO: add commands for the baseline, RAG, and evaluation stages as they are built]

## Status

This project is in progress. Current stage: data collection, evaluation question set, and baselines.