# 🎯 Cross-Sell Timing Analyzer

**A sentiment-driven cross-sell decision engine for e-commerce chatbots.**

Most chatbots blindly push product recommendations. This tool analyzes customer sentiment in real-time during a support conversation and answers the question every CX product manager cares about: **"Is this the right moment to cross-sell, or will it backfire?"**

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![HuggingFace](https://img.shields.io/badge/ML-HuggingFace%20Transformers-yellow)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## Why This Exists

In production chatbots, cross-sell timing is critical:

- **Too early** (customer is still frustrated) → cross-sell feels tone-deaf, damages trust
- **Too late** (conversation is ending) → missed revenue opportunity
- **Just right** (issue resolved, sentiment is positive) → 3-5x higher conversion rate

This project demonstrates how a simple ML sentiment model + product logic layer can make that timing decision automatically.

## What It Does

1. **Loads chat conversations** (sample e-commerce data included, or upload your own)
2. **Runs sentiment analysis** on each message using HuggingFace DistilBERT
3. **Plots the sentiment curve** across the conversation timeline
4. **Identifies cross-sell windows** — green (sell), yellow (wait), red (hold back)
5. **Validates against actual outcomes** — did the customer accept or reject the pitch?

### The Product Logic Layer

The ML model gives you a sentiment score (-1 to +1). But the **product decision** is what matters:

| Sentiment Score | Decision | Reasoning |
|---|---|---|
| ≥ 0.3 | ✅ Cross-sell | Customer is positive. Safe to suggest products. |
| -0.2 to 0.3 | ⏳ Wait | Neutral/mixed. Resolve query first. |
| ≤ -0.2 | 🛑 Hold back | Customer is frustrated. Focus on resolution. |

These thresholds are configurable in the UI — because different brands and contexts need different sensitivity.

## Sample Dataset

Includes 10 realistic multi-brand e-commerce conversations spanning:

- **Croma** — order tracking, service complaints
- **BigBasket** — delivery issues, positive feedback
- **1mg** — wrong medication, damaged packaging
- **CliQ** — payment disputes, happy customers
- **IHCL** — hotel booking changes, upsell opportunities

Each conversation includes cross-sell attempts with outcomes (accepted/rejected), so you can validate whether the model's recommendation aligns with real customer behavior.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/crosssell-sentiment-analyzer.git
cd crosssell-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`. First load downloads the DistilBERT model (~260MB, cached after).

## Using Your Own Data

Prepare a CSV with these columns:

| Column | Required | Description |
|---|---|---|
| conversation_id | ✅ | Unique ID per conversation |
| message_number | ✅ | Sequential message order |
| speaker | ✅ | "customer" or "agent" |
| message | ✅ | The actual message text |
| category | Optional | Label like "complaint", "cross_sell_attempt" |
| brand | Optional | Brand name for multi-brand analysis |

Upload via the sidebar in the app.

### Using a Kaggle Dataset

Good datasets to try:

- [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — real support conversations
- [Bitext Customer Support Dataset](https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset) — chatbot training data
- [Customer Service Chat Data 30k](https://www.kaggle.com/datasets/aimack/customer-service-chat-data-30k-rows) — large chat corpus

You'll need to rename columns to match the required format above.

## Deploy to Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — it handles dependencies automatically

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Sentiment Model | HuggingFace DistilBERT (SST-2) | Fast, accurate, free, no API key needed |
| App Framework | Streamlit | Fastest path from script to hosted app |
| Visualization | Plotly | Interactive charts with hover data |
| Data | Pandas | Standard for tabular data processing |

## How the ML Works

The sentiment model is **DistilBERT fine-tuned on SST-2** (Stanford Sentiment Treebank):

- **Architecture:** 6-layer Transformer (distilled from BERT-base)
- **Training data:** 67K movie review sentences labeled positive/negative
- **Accuracy:** ~91% on SST-2 benchmark
- **Inference:** ~50ms per message on CPU

The model outputs confidence scores for POSITIVE and NEGATIVE. The app converts this to a continuous score: `score = P(positive) - P(negative)`, giving a range from -1.0 (very negative) to +1.0 (very positive).

**Limitation:** The model is trained on English movie reviews, not customer support conversations. In production, you'd fine-tune on your own labeled support data for better accuracy. This project uses the pre-trained model as a proof-of-concept.

## Project Structure

```
crosssell-sentiment-analyzer/
├── app.py                    # Main Streamlit application
├── data/
│   └── sample_chats.csv      # Sample e-commerce chat dataset
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## License

MIT

## Author

**Sandhya Godavarthy** — [gsandhya.com](https://gsandhya.com)

Product Manager with 13+ years in AI/ML product delivery. Built production chatbots and email classification systems processing 17,000+ monthly queries across multiple e-commerce brands. This project demonstrates the product thinking layer that sits on top of ML models — the part that turns a sentiment score into a business decision.
