let's all put our ai usage here

# AI Usage Log

## Aarushi Jain

Tool: Claude (Anthropic)

### How I used AI

I used Claude as a coding assistant while setting up the repository and data pipeline. I chose the dataset and project scope with my team, ran all code myself, checked the output at each step, and fixed issues before committing. 

- Repository setup: help with the folder structure, .gitignore, and requirements.txt
- README.md: drafting the setup and reproducibility instructions, which I updated as our project plan changed
- src/download_data.py and run.py: help writing the script that downloads recall and complaint data from the public NHTSA API. I ran it and verified the downloaded record counts.
- notebooks/view_recalls.py: help writing a small script to print recall records in a readable format
- data/processed/questions.csv: I used the viewer script to read the 2019 Honda CR-V recall records, and used Claude to help turn them into evaluation questions with reference answers and campaign IDs. I checked the file loads correctly.
- Report: help drafting and revising the abstract to match our team's scope, and formatting the report in LaTeX

### Prompts

1. "ok lets write" (README draft)
2. "lets switch the dataset to this: https://www.nhtsa.gov/nhtsa-datasets-and-apis"
3. "what is this for" (explaining requirements.txt, .gitignore, and the virtual environment)
4. "i didnt paste the download script this is my last step so far" (download script)
5. [Shared recall records from terminal output] (drafting evaluation questions)
6. "ok also lets complete the download script we have" (run.py)
7. "lets chnahge the readme while we are at it. also give a title"
8. "ok update my abstract to match my teams new porject plan"
9. "this is the templete given give me the updated latex code"

### Where I changed the AI's suggestions

- I switched the dataset from the AI's suggestion (99P Labs blog posts) to NHTSA recall data.
- The AI's first abstract described a three-system project that did not match our team's plan, so I had it revised to match our literature review.

---

## Ryan Tang

I used ChatGPT to proofread the literature review. I used CODEX to implement some parts of the baseline LLM and debug. For the most part, I was satisfied with generated responses, so I haven't yet needed to change any of AI's suggestions.

## Prompts (ChatGPT)

Attached is a draft of a literature review for my Capstone project comparing Retrieval Augmented Generation (RAG) to parametric LLM. Proofread for grammar and tone, and help properly format the attached sources.

## Prompts (Codex)

### Prompt 1

I am currently responsible for implementing the LLM-only baseline for our capstone. I want to understand the implementation and be able to explain every major design decision. I'm working on a new branch 'baseline,' but we should base the implementation on the stuff in the branch 'rag'.

Please help me implement a minimal, reproducible LLM-only baseline.

Requirements:
1. Reuse the existing Groq generation approach and openai/gpt-oss-20b model.
2. Create a reusable src/llm.py module for model inference.
3. Create src/baseline.py to run questions through the LLM without retrieval.
4. Read evaluation questions from a JSON file.
5. Save responses, model settings, latency, and token usage to results/.
6. Use environment variables for API credentials.
7. Make model generation settings explicit.
8. Avoid any Ollama, embeddings, RAG retrieval, or NHTSA data dependencies in the baseline.
9. Add a minimal test using a mocked Groq client that does not make paid API calls.
10. Provide clear instructions for running the baseline.
Important:
- Do not modify the existing RAG notebook or other teammates' files.
- Explain the proposed implementation before writing code.
- Implement incrementally so I can review each file and understand it.
- Avoid adding unnecessary dependencies or frameworks.
- Do not run paid API calls without my approval.
- Do not commit or push changes automatically.
Start by showing me the design of the two modules, the evaluation question format, and the proposed command to run the baseline. Then wait for my approval before making changes.

Also, you'll probably need to update the environment and requirements.txt to include the Ollama stuff. Please install all necessary dependencies onto my computer.

### Prompt 2

I've approved src/llm.py and want to test it with a real Groq inference request.

Please create a minimal scripts/smoke_test_llm.py that:
1. Imports and uses our existing GroqLLM and GenerationSettings classes.
2. Sends one simple automotive recall-related question.
3. Prints the response, model name, latency, and token usage.
4. Reports API failures clearly without exposing credentials.
5. Uses the GROQ_API_KEY environment variable.
6. Does not use retrieval, NHTSA files, embeddings, or Ollama.
Use the actual interfaces from src/llm.py, not guessed method signatures.

Do not modify src/llm.py or the RAG notebook. Do not make the API request yourself. Show me the script and the exact command needed to run it locally.

### Prompt 3

Our first live Groq smoke test reached the API successfully, but returned empty answer text.

Results:
- Model: openai/gpt-oss-20b
- Latency: 2.384 seconds
- Prompt tokens: 84
- Completion tokens: 1024 (exactly our configured limit)
- Answer: empty
I think GPT-OSS exhausted the completion-token budget on reasoning before generating its final answer.

Please:
1. Inspect how src/llm.py handles GPT-OSS reasoning and response content.
2. Check and report the response's finish_reason and available reasoning-token usage.
3. Add configurable reasoning_effort="low" and increase max_completion_tokens to 4096 for our next test.
4. Ensure a response with empty final content is explicitly identified, rather than treated as a successful answer.
5. Preserve our structured error handling and existing tests.
Make only the necessary changes, explain them, and don't make API calls yourself.

### Prompt 4

I am on branch 'baseline' and want to merge into main. Please stage the right commits for me to merge into main, making sure everything stays intact and there are no conflicts. For right now, I am not concerned with keeping all code intact and making sure code isn't redundant.

---

## Shawn Cheng

[Add your AI usage here.]

---

## Henry Liu

Tools: Codex, Claude (Anthropic)

### How I used AI
- Environment and imports: Codex wrote the imports cell; I created the virtual environment and selected the kernel myself.
- Data loading: Codex wrote the 29-column schema and loader from NHTSA's `RCL.txt`.
- CAMPNO counts, report-year chart, missing-value and placeholder checks: Codex wrote the code; Claude suggested the checks and an earlier version of the code.
- Duplicate-record checks and dataset overview: Codex wrote the cells and tables.
- Advisory-flag cells (`DO_NOT_DRIVE`, `PARK_OUTSIDE`): code suggested by Claude.
- Notebook structure: Claude suggested the section order and which cells to move; I reorganized the notebook.
- Codex also drafted explanatory Markdown for the dataset overview and duplicate-check tables. Claude suggested commentary wording. I reviewed and edited the interpretations

### Prompts
Codex: 
- Help me edit the ipynb so that I have the proper imports
- Help me load the dataset
- Help me load the dataset using the RCL.txt file given from the NHTSA stating what values are contained within the recall dataset. Since the txt file with the data doesn't have a header, we would need to populate it with the values from this file
- Now that we have the data loaded, provide me code blocks in the following order:
    1. The number of unique NHTSA Recall Numbers (CAMPNO) and how many are null (Null values are generally unusable for us in this case since a recall should be associated with a year)
    2. A graph showing the distribution of the recall years (taking the first 4 digits of each RCDATE) so that we can better understand the years that the dataset actually spans and the distribution of recalls
    3. A check on the number of missing values and placeholders: YEARTXT uses 9999 for unknown or N/A (common for equipment and tire recalls). Check for other sentinel values.
- Add cells to my notebook that check for duplicate records in recalls_df, with the
results in tables. Don't modify recalls_df.
    1. Summary table with one row per check: total rows, unique RECORD_ID, duplicated RECORD_ID rows, fully duplicated rows (all 29 columns), unique CAMPNO, and rows duplicated on the key [CAMPNO, MAKETXT, MODELTXT, YEARTXT, COMPNAME]. Include count and % of total rows for each.
    2. Rows-per-CAMPNO table (describe with 50/90/99th percentiles) plus the top 10 CAMPNOs by row count.
    3. If any fully duplicated rows or duplicated keys exist, show a sample of up to 10 (sorted so duplicates sit together). If none, say so. Add a short markdown cell after each table saying what the output shows and what it does not establish. Treat NA values as equal when detecting duplicates.
- Help me write the dataset overview, where you include 
    1. Dataset shape: rows and columns so that we can understand the scope that we are working with so we don't run into memory issues later
    2. Column inventory: one table with each column’s loaded dtype, non-null count, nunique, and the intended type (identifier, date, numeric, categorical, free text)

Claude:
- What I should be looking for in EDA
- For duplicate information would it be better to have a table showing the nunique or a chart?
- Tell me exactly which cells need to move where (Asking for advice on how to reorder sections so that labels are accurate)


### Testing
- Restarted the kernel and ran all cells to see if code actually ran properly
- Cross checked listed information with NHTSA's provided data information file for recalls

### Where I changed the AI's suggestions
- Rewrote the notebook’s markdown commentary where I disagreed with the AI’s interpretation or saw a bigger takeaway elsewhere, such as the high missingness of RPNO
- Chose the section order and heading names myself, using Claude’s suggestions as a starting point.