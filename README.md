# Student Support and Course Feedback Assistant

AI for Everyone, Project 01. A classroom prototype built with Streamlit.
The support flag is a modelling exercise on public practice data. It is **not** a decision about a real student, and a teacher must review every suggestion.

## What the app does (5 tabs)
1. **Class overview and support prediction**: Decision Tree on G1, G2, studytime, failures, schoolsup, famsup, internet.
2. **Learning profiles**: KMeans with 3 groups (G1, G2, studytime, failures).
3. **Feedback digit**: small CNN reads one handwritten digit; the teacher confirms it.
4. **Course comments**: NLTK VADER sentiment and keyword topic rules (English only).
5. **Teaching summary**: Offline template by default; optional Groq Free Plan summary.

The app only loads saved models. It never trains when you click.

## Repository layout
```
app.py
requirements.txt
README.md
.gitignore
data/            main.csv, feedback.csv, example_inputs.csv, meta.json, topic_keywords.json, prompt_test_cases.json
models/          supervised.joblib, clusters.joblib, images.keras, group_descriptions.json
                 (optional: tabular_metrics.json, cnn_metrics.json, nlp_metrics.json)
nltk_data/       sentiment/vader_lexicon.zip   (offline lexicon)
sample_images/   supplied digit images for the demo
source_notices/  attribution and licences (keep these)
```
Do not upload the full `images.npz`, ZIP archives or any API key.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (free)
1. Upload the files above to a free GitHub repository.
2. On Streamlit Community Cloud, connect the repository, pick the branch and `app.py`, and deploy.
3. Optional AI summary: in the app's **Secrets** panel add
```
GROQ_API_KEY = "your key"
GROQ_MODEL = "openai/gpt-oss-20b"
```
Without a key the app stays in Offline mode and everything else works.

## Limits
- Small public dataset; 60 test rows, so metrics can move with small changes.
- MNIST digits are clean; real handwriting or scanned forms may fail.
- VADER finds tone only, and struggles with mixed opinions, sarcasm and negation.
- Groq free quotas can return rate-limit errors. The app then offers the Offline summary.

## Data sources
UCI Student Performance (mathematics), MNIST digits, authored classroom comments, NLTK VADER lexicon. See `source_notices/`.

## Team contributions
(Write each student's work here.)
