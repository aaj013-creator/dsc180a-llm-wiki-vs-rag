# LLM-Wiki vs RAG: Organizing Knowledge for AI Agents

DSC 180A Capstone, Quarter 1 Project
Mentor: Ryan (99P Labs)
Team: Aarushi Jain, Ryan Tang, Shawn Cheng, Henry Liu

## Overview

Large language models can answer many questions, but they need access to external knowledge to answer questions about a specific domain. This project compares three ways of giving an LLM access to a knowledge corpus of NHTSA vehicle safety recalls and owner complaints for Honda vehicles:

1. LLM only: no access to the corpus (baseline)
2. Naive RAG: records are chunked and embedded, and the most similar chunks are retrieved for each question
3. LLM-wiki: an LLM agent reads the records and maintains organized wiki pages, then answers questions from those pages

We evaluate each system on two tasks:

- Single-record lookup: questions answered by one recall or complaint
- Cross-record synthesis: questions that require combining information across many recalls and complaints

## Evaluation Scorecard

| Metric | Description |
| --- | --- |
| Answer correctness | Scored by an LLM judge, spot-checked by hand |
| Groundedness | Whether the answer cites the correct records |
| Abstention | Whether the system says it does not know on unanswerable questions |
| Cost | Tokens used per question |

## Repository Structure

```
.
├── data/
│   ├── raw/          # Raw NHTSA JSON responses
│   └── processed/    # Cleaned records and question set
├── notebooks/        # Exploration and analysis
├── results/          # Scorecard outputs
├── src/              # Source code (loaders, baselines, evaluation)
├── run.py            # Runs the full pipeline
├── requirements.txt  # Python dependencies
└── README.md
```

## Getting the Data

Data comes from the public NHTSA API (https://www.nhtsa.gov/nhtsa-datasets-and-apis). No API key is required.

Current scope:

- Make: Honda
- Models: Civic, Accord, CR-V
- Model years: 2018 to 2022
- Sources: recalls and owner complaints

To download the data into `data/raw/`, run:

```
python run.py data
```

## Setup

Requires Python 3.10 or newer.

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

```
python run.py data
```

[TO DO: add commands for each stage as they are built]

## Status

This project is in progress. Current stage: data collection and baselines.