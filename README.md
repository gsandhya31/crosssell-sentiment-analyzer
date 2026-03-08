# 🎯 Cross-Sell Timing Analyzer

**A sentiment-driven cross-sell decision engine for e-commerce chatbots.**

Most chatbots blindly push product recommendations. This tool analyzes customer sentiment in real-time during a support conversation and answers the question every CX product manager cares about: **"Is this the right moment to cross-sell, or will it backfire?"**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![HuggingFace](https://img.shields.io/badge/ML-HuggingFace%20Transformers-yellow)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

**[Live Demo](https://crosssell-sentiment-analyzer-gsandhya.streamlit.app/)** · **[GitHub](https://github.com/gsandhya31/crosssell-sentiment-analyzer)**

---

## Why This Exists

In production chatbots, cross-sell timing is critical:

- **Too early** (customer is still frustrated) → cross-sell feels tone-deaf, damages trust
- **Too late** (conversation is ending) → missed revenue opportunity
- **Just right** (issue resolved, sentiment is positive) → 3-5x higher conversion rate

This project demonstrates how a simple ML sentiment model + product logic layer can make that timing decision automatically.

## What It Does

1. **Loads chat conversations** (sample e-commerce data included, or upload your own)
2. **Runs sentiment analysis** on each message using HuggingFace Twitter RoBERTa
3. **Plots the sentiment curve** across the conversation timeline
4. **Identifies cross-sell windows** — green (sell), yellow (wait), red (hold back)
5. **Validates against actual outcomes** — did the customer accept or reject the pitch?
6. **Cross-sell performance dashboard** — success rate by sentiment zone, proving that timing drives conversion

### The Product Logic Layer

The ML model gives you a sentiment score (-1 to +1). But the **product decision** is what matters:

| Sentiment Score | Decision | Reasoning |
|---|---|---|
| ≥ 0.3 | ✅ Cross-sell | Customer is positive. Safe to suggest products. |
| -0.2 to 0.3 | ⏳ Wait | Neutral/mixed. Resolve query first. |
| ≤ -0.2 | 🛑 Hold back | Customer is frustrated. Focus on resolution. |

These thresholds are configurable in the UI — because different brands and contexts need different sensitivity.

### Cross-Sell Performance Dashboard

The dashboard aggregates all conversations and answers the business question: **does cross-selling at the right sentiment moment actually convert better?**

It shows:
- Customer messages by sentiment zone (green/yellow/red)
- Cross-sell attempts per zone with acceptance vs rejection counts
- Success rate comparison across zones
- Auto-generated insight (e.g., "Green zone converts at 4x the rate of red zone")
- Detailed log of every cross-sell attempt with pre-pitch sentiment score

## Key Product Decisions

### Why Twitter RoBERTa over DistilBERT

The initial version used DistilBERT fine-tuned on SST-2 (movie reviews, 2-class: positive/negative). This had a critical flaw — **no neutral class**. A casual message like "what are you up to?" scored 0.98 positive, which would have triggered a false cross-sell recommendation.

We switched to `cardiffnlp/twitter-roberta-base-sentiment-latest` (3-class: positive/neutral/negative) because:
- Customer support language is casual and short, closer to tweets than movie reviews
- Detecting "neutral" is critical — you don't want a routine "okay" triggering a sales pitch
- The model was trained on 124M tweets, so it handles slang, abbreviations, and informal tone

### Why per-message scoring (and its limitations)

The current implementation scores each customer message independently. This is a deliberate starting point for the MVP, but **not production-ready**. The problem: a customer who was furious 2 minutes ago and now says "okay thanks" gets scored as positive. That's grudging acceptance, not genuine satisfaction.

**The production-ready approach** is to trigger cross-sell on **resolved positive** — a specific pattern where: (1) the customer had a problem, (2) the agent resolved it, (3) the customer's sentiment genuinely shifted positive over multiple messages. This requires conversation-level context, not just per-message scoring.

See "Improvements for Production" below for the architectural approach.

## Sample Dataset

Includes 10 realistic multi-brand e-commerce conversations spanning:

- **Croma** — order tracking, service complaints
- **BigBasket** — delivery issues, positive feedback
- **1mg** — wrong medication, damaged packaging
- **CliQ** — payment disputes, happy customers
- **IHCL** — hotel booking changes, upsell opportunities

Each conversation includes **outcome labels** (`cross_sell_success` / `cross_sell_failure` in the category column) so you can validate whether the model's recommendation aligns with real customer behavior.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/gsandhya31/crosssell-sentiment-analyzer.git
cd crosssell-sentiment-analyzer

# Create virtual environment (Python 3.11+)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. First load downloads the Twitter RoBERTa model (~500MB, cached after).

## Using Your Own Data

Prepare a CSV with these columns:

| Column | Required | Description |
|---|---|---|
| conversation_id | ✅ | Unique ID per conversation |
| message_number | ✅ | Sequential message order |
| speaker | ✅ | "customer" or "agent" |
| message | ✅ | The actual message text |
| category | Optional | Label like "complaint", "cross_sell_attempt", "cross_sell_success" |
| brand | Optional | Brand name for multi-brand analysis |

Upload via the sidebar in the app.

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Sentiment Model | Twitter RoBERTa (3-class) | Handles casual language, detects neutral, trained on 124M tweets |
| App Framework | Streamlit | Fastest path from script to hosted app |
| Visualization | Plotly | Interactive charts with hover data |
| Data | Pandas | Standard for tabular data processing |

## How the ML Works

The sentiment model is **Twitter RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment-latest`):

- **Architecture:** RoBERTa base (125M parameters) — a robustly optimized version of BERT
- **Pre-training:** 124 million tweets
- **Fine-tuning:** TweetEval sentiment benchmark (3-class: positive, neutral, negative)
- **Output:** Three confidence scores that sum to 1.0

The app converts these to a single continuous score: `score = P(positive) - P(negative)`, giving a range from -1.0 (very negative) to +1.0 (very positive). When a message is neutral, both positive and negative scores are low, so the score lands near 0.0 — correctly placing it in the "wait" zone.

## Improvements for Production

### 1. Conversation-level sentiment (not per-message)

Replace independent per-message scoring with a **rolling weighted average** of the last N customer messages. Recent messages weighted higher (50% / 30% / 20%). This prevents a single "okay thanks" after a complaint from triggering a cross-sell.

The ideal production trigger: **resolved positive** — detect the pattern where sentiment trajectory went negative → agent intervened → sentiment recovered to positive over 2-3 messages. This is a genuine recovery, not grudging acceptance.

### 2. Cascade architecture for scale

Don't call an expensive model for every message:
- **Tier 1 (every message):** Cheap, fast model (Twitter RoBERTa, ~50ms, free) handles obvious green/red cases
- **Tier 2 (yellow zone only):** LLM call with full conversation context for ambiguous cases (~15-20% of conversations)
- **Tier 3 (batch, post-conversation):** LLM auto-labels cross-sell attempts and outcomes for the dashboard

### 3. Auto-detection of cross-sell attempts

Currently the dataset is manually labelled. In production, use keyword rules or an LLM classifier to automatically detect when an agent attempted a cross-sell and whether it was accepted.

### 4. Fine-tune on domain data

The Twitter RoBERTa model is general-purpose. Fine-tuning on labelled customer support conversations from the specific domain would improve accuracy significantly.

## Project Structure

```
crosssell-sentiment-analyzer/
├── app.py                    # Main Streamlit application
├── data/
│   └── sample_chats.csv      # Sample e-commerce chat dataset (10 conversations, 67 messages)
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## License

MIT

## Author

**Sandhya Godavarthy** — [gsandhya.com](https://gsandhya.com)

Product Manager with 13+ years in AI/ML product delivery. Built production chatbots and email classification systems processing 17,000+ monthly queries across multiple e-commerce brands. This project demonstrates the product thinking layer that sits on top of ML models — the part that turns a sentiment score into a business decision.
