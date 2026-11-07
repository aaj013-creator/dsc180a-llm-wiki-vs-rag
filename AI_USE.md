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

[Add your AI usage here.]

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
- Rewrote the notebook’s markdown commentary where I disagreed with the AI’s interpretation or saw a bigger takeaway elsewhere, such as the high missinness of RPNO
- Chose the section order and heading names myself, using Claude’s suggestions as a starting point.