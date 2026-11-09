# Customer Review Intelligence | NLP & LLM-Powered Product Analytics

An end-to-end product analytics case study using **3,090 Ownwell customer reviews** to identify recurring complaints, quantify customer friction, and surface opportunities to improve onboarding, billing transparency, and customer support.

The project combines **VADER sentiment analysis**, **bag-of-words analysis**, **sentence-transformer embeddings**, and **LLM-based structured claim extraction**. Results are visualized with Plotly and Matplotlib and exported as CSV files for further analysis.

> **Project scope:** Exploratory analysis of customer reviews. The complaint labels and risk scores are model-generated interpretations, not verified findings of wrongdoing or measured customer churn.

## Key Results

| Metric | Result |
| --- | ---: |
| Customer reviews analyzed | 3,090 |
| Five-star reviews | 89.4% |
| One- and two-star reviews | 218 (7.1%) |
| Negative reviews about savings outcomes | 94 (43.1% of negative reviews) |
| Negative reviews about support responsiveness | 75 (34.4% of negative reviews) |
| Share of negative reviews in those two categories | 77.5% |

### LLM-classified complaint breakdown

| Complaint category | Reviews | Share of negative reviews | Mean model-assigned risk score (1–3) |
| --- | ---: | ---: | ---: |
| Dissatisfied savings | 94 | 43.1% | 2.89 |
| Unresponsive support | 75 | 34.4% | 2.85 |
| Unexpected auto-renewal | 24 | 11.0% | 2.96 |
| Quote-to-invoice misunderstanding | 19 | 8.7% | 2.95 |
| Homestead exemption fee dispute | 6 | 2.8% | 2.83 |
| **Total** | **218** | **100%** | — |

The analysis suggests that **savings expectations and support accessibility** are the largest complaint themes by volume. **Auto-renewal and quote-to-invoice misunderstandings** received the highest average *LLM-assigned* risk scores and may merit closer investigation into customer communication and billing experiences.

## Problem Statement

Aggregate star ratings can hide important customer experience issues. The goal of this project was to move beyond an overall satisfaction score and answer three product questions:

1. What are customers actually complaining about?
2. Which themes occur most frequently, and which warrant prioritization?
3. How can unstructured review text be converted into actionable product insights?

## Methodology

### 1. Data preparation and sentiment analysis

- Loaded review data from `ownwell_reviews.csv` using Pandas.
- Extracted numeric star ratings and handled missing review text.
- Used **NLTK VADER** to calculate positive, neutral, negative, and compound sentiment scores.
- Compared language-based sentiment against numeric star ratings.
- Applied keyword-based business-theme mapping as an initial interpretability layer.

### 2. Complaint keyword discovery

- Filtered **1- and 2-star reviews**.
- Applied scikit-learn `CountVectorizer` with English and domain-specific stop words.
- Identified frequent **unigrams and bigrams** to explore recurring complaint language.
- Visualized the top terms using Matplotlib.

### 3. Semantic complaint matching

- Encoded negative reviews and four candidate complaint hypotheses with **`sentence-transformers/all-MiniLM-L6-v2`**.
- Calculated cosine similarity between review and hypothesis embeddings.
- Assigned each review its highest-similarity candidate hypothesis and recorded a similarity score.

This stage provides exploratory semantic matches, not ground-truth complaint labels or calibrated probabilities.

### 4. Structured LLM claim extraction

- Called **`openai/gpt-4o-mini` through OpenRouter** using the OpenAI Python SDK.
- Classified each negative review into one of five predefined complaint categories.
- Requested a structured JSON response containing an `exact_claim` and an integer `churn_risk_score` from 1 to 3.
- Aggregated claim counts and average assigned risk scores by category.

The notebook first tests this workflow on 10 reviews, then runs classification across the full negative-review subset.

### 5. Visualization and export

- Built interactive Plotly charts for sentiment-versus-rating distributions, complaint categories, and average assigned risk scores.
- Exported the aggregate sentiment summary and review-level LLM analysis to CSV.

## Tech Stack

| Category | Tools |
| --- | --- |
| Language and environment | Python, Jupyter Notebook, Google Colab |
| Data manipulation | Pandas, NumPy |
| Sentiment analysis | NLTK VADER |
| Text feature extraction | scikit-learn `CountVectorizer` |
| Semantic similarity | Sentence Transformers (`all-MiniLM-L6-v2`), cosine similarity |
| LLM classification | GPT-4o mini via OpenRouter, OpenAI Python SDK |
| Visualization | Plotly Express, Matplotlib |

## Getting Started

### Prerequisites

- Python 3.10+ or Google Colab
- The input dataset, `ownwell_reviews.csv`
- An [OpenRouter](https://openrouter.ai/) API key for the LLM extraction steps
- Internet access to download the VADER lexicon and Sentence Transformer model

### Installation

Clone this repository, install the dependencies, and open the notebook:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy nltk plotly matplotlib scikit-learn sentence-transformers openai notebook
jupyter notebook
```

Rename the uploaded notebook to `ownwell_customer_review_analysis.ipynb`, or open it using its current filename.

### Configure the dataset

The notebook currently reads the dataset from:

```python
file_path = '/content/ownwell_reviews.csv'
```

In **Google Colab**, upload `ownwell_reviews.csv` to `/content/`. For local use, change `file_path` to the dataset's location.

The code expects at least these CSV columns:

| Column | Description |
| --- | --- |
| `review_id` | Review identifier |
| `rating` | Star rating as text (e.g., `5 stars`) |
| `review_text` | Customer review text |

### Configure the API key

In **Google Colab**, create a secret named `openrouter_api_key`. Alternatively, set an environment variable with that same name before running the notebook:

```bash
export openrouter_api_key="YOUR_OPENROUTER_API_KEY"
```

For Windows PowerShell:

```powershell
$env:openrouter_api_key="YOUR_OPENROUTER_API_KEY"
```

**Never commit API keys to GitHub.** The current notebook imports `google.colab.userdata`, so to run the OpenRouter section locally, remove or adapt that import and use `os.getenv('openrouter_api_key')` directly.

### Run the analysis

Execute the notebook cells in order. The pipeline performs sentiment scoring, complaint keyword analysis, semantic matching, LLM classification, aggregation, and CSV export.

> **Cost note:** The complete LLM classification stage makes one API request per negative review. Charges and rate limits depend on the OpenRouter account and selected model.

## Outputs

| Output | Description |
| --- | --- |
| `ownwell_sentiment_summary.csv` | Aggregated review count, average star rating, and average VADER sentiment by business theme |
| `ownwell_full_openrouter_analysis.csv` | Review-level dataset with sentiment, complaint labels, and model-assigned risk scores |
| Plotly charts | Star-rating/sentiment comparison and claim-category analysis |
| Matplotlib charts | Most frequent complaint unigrams and bigrams |

Generated CSV files are written to the notebook's working directory. They are not required to be committed to the repository.

## Product Insights and Recommendations

The following are **hypotheses informed by review analysis**, not confirmed causal explanations:

1. **Make expected savings clearer.** Set customer expectations around possible property-tax outcomes, fees, and uncertainty before enrollment.
2. **Improve support accessibility.** Investigate response times and escalation paths for customers disputing outcomes or charges.
3. **Clarify renewal terms.** Review how annual renewals and customer obligations are communicated during onboarding and before renewal.
4. **Reduce quote-versus-contract confusion.** Test whether clearer confirmation screens and explicit fee disclosures reduce misunderstanding.
5. **Validate complaint severity with operational data.** Compare model-labeled themes against cancellations, billing disputes, support tickets, and chargebacks before treating risk scores as predictive.

## Limitations

- **Review selection bias:** Public reviews may not represent the full customer base.
- **Unverified labels:** Complaint categories are assigned by heuristics, embedding similarity, or an LLM; there is no annotated validation set or reported classification accuracy.
- **Risk is not churn prediction:** The 1–3 score is generated from review wording. It is not trained or validated against observed churn, fraud, chargebacks, or legal outcomes.
- **Single-label classification:** Each negative review is assigned one category even if multiple issues are present.
- **Potential API errors:** The notebook defaults to an error label and risk score of 1 when an LLM request fails; inspect these records before interpreting aggregates.
- **Reproducibility:** Exact outputs depend on the input dataset, model responses, and API availability.

## Future Improvements

- Create a manually labeled validation sample and measure precision, recall, and agreement.
- Add multi-label classification for reviews containing several complaints.
- Validate risk scores against actual retention and support outcomes.
- Track themes over time to detect emerging issues.
- Add retry logic, caching, batching, and structured validation for API responses.
- Turn recurring analyses into a product-facing dashboard.

## Repository Note

This repository documents an **independent exploratory product analytics case study**. Customer reviews describe individual experiences; classification results should not be interpreted as verified factual findings about Ownwell's conduct. This project does not imply affiliation with or endorsement by Ownwell.
