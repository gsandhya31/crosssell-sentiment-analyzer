"""
Cross-Sell Timing Analyzer
==========================
A sentiment-driven cross-sell decision engine for e-commerce chatbots.

Uses Hugging Face sentiment analysis to analyze customer chat conversations 
and identify optimal moments for cross-selling vs moments to hold back.

Built as a portfolio project demonstrating ML API integration with product logic.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline
import numpy as np
import os

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Sell Timing Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .sell-window {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
    }
    .hold-window {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
    }
    .neutral-window {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 15px;
        border-radius: 0 8px 8px 0;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_sentiment_model():
    """Load HuggingFace sentiment analysis pipeline. Cached across reruns."""
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        return_all_scores=True
    )


def get_sentiment_score(text, analyzer):
    """
    Returns a sentiment score between -1 (very negative) and +1 (very positive).
    Uses the HuggingFace model's confidence scores to create a continuous scale.
    Handles different output shapes from the transformers pipeline.
    """
    raw = analyzer(text[:512])  # Truncate to model's max length

    # Normalize to a flat list of {label, score} dicts
    candidates = []
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, list):
            candidates = [d for d in first if isinstance(d, dict)]
        elif isinstance(first, dict):
            candidates = [d for d in raw if isinstance(d, dict)]

    if not candidates:
        return 0.0

    scores = {
        item.get("label"): float(item.get("score", 0.0))
        for item in candidates
        if isinstance(item, dict) and item.get("label") is not None
    }

    # Convert to -1 to +1 scale: positive score - negative score
    # Twitter RoBERTa model uses lowercase labels: positive, negative, neutral
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def classify_crosssell_window(score, threshold_sell=0.3, threshold_hold=-0.2):
    """
    Product logic layer: decide cross-sell action based on sentiment.
    
    This is the PM decision — not just "is it positive?" but 
    "is it positive ENOUGH to risk a sales pitch without annoying the customer?"
    
    Args:
        score: sentiment score (-1 to +1)
        threshold_sell: minimum score to attempt cross-sell (default 0.3)
        threshold_hold: below this, absolutely do NOT cross-sell (default -0.2)
    
    Returns:
        tuple: (decision, confidence, reasoning)
    """
    if score >= threshold_sell:
        confidence = min((score - threshold_sell) / (1 - threshold_sell) * 100, 100)
        return ("✅ CROSS-SELL", confidence, 
                "Customer sentiment is positive. Safe to suggest related products.")
    elif score <= threshold_hold:
        confidence = min((threshold_hold - score) / (1 + threshold_hold) * 100, 100)
        return ("🛑 HOLD BACK", confidence,
                "Customer is frustrated. Focus on resolution, not selling.")
    else:
        return ("⏳ WAIT", 50,
                "Sentiment is neutral/mixed. Resolve the query first, then reassess.")


def analyze_conversation(df, analyzer, sell_threshold, hold_threshold):
    """Analyze each message in a conversation and add sentiment + cross-sell decision."""
    results = []
    for _, row in df.iterrows():
        score = get_sentiment_score(row['message'], analyzer)
        decision, confidence, reasoning = classify_crosssell_window(
            score, sell_threshold, hold_threshold
        )
        results.append({
            **row.to_dict(),
            'sentiment_score': round(score, 3),
            'crosssell_decision': decision,
            'confidence': round(confidence, 1),
            'reasoning': reasoning
        })
    return pd.DataFrame(results)


def plot_sentiment_curve(conv_df):
    """Plot the sentiment trajectory of a conversation with cross-sell zones."""
    fig = go.Figure()

    # Add colored zones
    fig.add_hrect(y0=0.3, y1=1.0, fillcolor="green", opacity=0.08,
                  annotation_text="Cross-sell zone", annotation_position="top left")
    fig.add_hrect(y0=-0.2, y1=0.3, fillcolor="yellow", opacity=0.08,
                  annotation_text="Wait zone", annotation_position="top left")
    fig.add_hrect(y0=-1.0, y1=-0.2, fillcolor="red", opacity=0.08,
                  annotation_text="Hold back zone", annotation_position="top left")

    # Customer messages
    cust_df = conv_df[conv_df['speaker'] == 'customer']
    fig.add_trace(go.Scatter(
        x=cust_df['message_number'],
        y=cust_df['sentiment_score'],
        mode='lines+markers',
        name='Customer Sentiment',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        hovertemplate=(
            '<b>Message %{x}</b><br>'
            'Sentiment: %{y:.2f}<br>'
            '<extra></extra>'
        )
    ))

    # Mark cross-sell attempts from agent
    agent_sell = conv_df[
        (conv_df['speaker'] == 'agent') & 
        (conv_df['category'].str.contains('cross_sell', na=False))
    ]
    if not agent_sell.empty:
        fig.add_trace(go.Scatter(
            x=agent_sell['message_number'],
            y=[0] * len(agent_sell),
            mode='markers',
            name='Cross-sell Attempted',
            marker=dict(size=14, symbol='star', color='gold', 
                       line=dict(width=2, color='orange')),
        ))

    fig.update_layout(
        title="Sentiment Trajectory & Cross-Sell Windows",
        xaxis_title="Message Number",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def render_chat_with_sentiment(conv_df):
    """Render the chat messages with color-coded sentiment indicators."""
    for _, row in conv_df.iterrows():
        speaker = row['speaker']
        score = row['sentiment_score']
        decision = row['crosssell_decision']
        
        if speaker == 'customer':
            # Color based on sentiment
            if score >= 0.3:
                css_class = "sell-window"
            elif score <= -0.2:
                css_class = "hold-window"
            else:
                css_class = "neutral-window"
            
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>👤 Customer</strong> (sentiment: {score:+.2f}) '
                f'<span style="float:right">{decision}</span><br>'
                f'{row["message"]}</div>',
                unsafe_allow_html=True
            )
        else:
            is_crosssell = 'cross_sell' in str(row.get('category', ''))
            prefix = "🏷️ " if is_crosssell else ""
            st.markdown(
                f'<div style="background:#e8eaf6; padding:10px 15px; '
                f'border-radius:8px; margin:5px 0;">'
                f'<strong>🤖 Agent</strong> {prefix}<br>'
                f'{row["message"]}</div>',
                unsafe_allow_html=True
            )


# ─── MAIN APP ───────────────────────────────────────────────────────────
def main():
    st.markdown('<p class="main-header">🎯 Cross-Sell Timing Analyzer</p>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Sentiment-driven cross-sell decisions for e-commerce chatbots — '
        'know WHEN to sell and when to hold back.'
        '</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ─── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Thresholds")
        sell_threshold = st.slider(
            "Cross-sell threshold",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            help="Minimum sentiment score to trigger cross-sell recommendation"
        )
        hold_threshold = st.slider(
            "Hold-back threshold",
            min_value=-1.0, max_value=0.0, value=-0.2, step=0.05,
            help="Below this score, suppress all cross-sell attempts"
        )
        
        st.markdown("---")
        st.subheader("📁 Data Source")
        data_source = st.radio(
            "Choose data source",
            ["Sample Dataset", "Upload Your Own CSV"]
        )
        
        uploaded_file = None
        if data_source == "Upload Your Own CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=['csv'],
                help="CSV must have columns: conversation_id, message_number, speaker, message"
            )
            st.markdown(
                "**Required columns:** conversation_id, message_number, "
                "speaker (customer/agent), message"
            )
        
        st.markdown("---")
        st.markdown(
            "**Built with:** Hugging Face Transformers · Streamlit · Plotly\n\n"
            "**Model:** DistilBERT (fine-tuned on SST-2)\n\n"
            "**Author:** [Sandhya Godavarthy](https://www.linkedin.com/in/sandhya-godavarthy-5072622b/)"
        )

    # ─── Load Data ──────────────────────────────────────────────────────
    if data_source == "Upload Your Own CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = ['conversation_id', 'message_number', 'speaker', 'message']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return
    else:
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "sample_chats.csv"))

    # ─── Load Model ─────────────────────────────────────────────────────
    with st.spinner("Loading sentiment model (first time may take ~30s)..."):
        analyzer = load_sentiment_model()

    # ─── Conversation Selector ──────────────────────────────────────────
    conversations = df['conversation_id'].unique()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_conv = st.selectbox(
            "Select Conversation",
            conversations,
            format_func=lambda x: f"{x} — {df[df['conversation_id']==x].iloc[0].get('brand', 'N/A')} | {df[df['conversation_id']==x].iloc[0].get('category', 'N/A')}"
        )
    
    conv_df = df[df['conversation_id'] == selected_conv].copy()
    
    with col2:
        brand = conv_df.iloc[0].get('brand', 'Unknown')
        category = conv_df.iloc[0].get('category', 'Unknown')
        msg_count = len(conv_df)
        st.markdown(
            f"**Brand:** {brand} &nbsp;|&nbsp; "
            f"**Category:** {category} &nbsp;|&nbsp; "
            f"**Messages:** {msg_count}"
        )

    # ─── Run Sentiment Analysis ─────────────────────────────────────────
    with st.spinner("Analyzing sentiment..."):
        results_df = analyze_conversation(conv_df, analyzer, sell_threshold, hold_threshold)

    # ─── Metrics Row ────────────────────────────────────────────────────
    customer_results = results_df[results_df['speaker'] == 'customer']
    avg_sentiment = customer_results['sentiment_score'].mean()
    final_sentiment = customer_results['sentiment_score'].iloc[-1] if len(customer_results) > 0 else 0
    sell_windows = len(customer_results[customer_results['crosssell_decision'].str.contains('CROSS-SELL')])
    hold_windows = len(customer_results[customer_results['crosssell_decision'].str.contains('HOLD')])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Sentiment", f"{avg_sentiment:+.2f}", 
              delta="Positive" if avg_sentiment > 0 else "Negative")
    m2.metric("Final Sentiment", f"{final_sentiment:+.2f}",
              delta="Positive" if final_sentiment > 0 else "Negative")
    m3.metric("Cross-sell Windows", f"{sell_windows}/{len(customer_results)}")
    m4.metric("Hold-back Moments", f"{hold_windows}/{len(customer_results)}")

    # ─── Sentiment Curve ────────────────────────────────────────────────
    st.plotly_chart(plot_sentiment_curve(results_df), use_container_width=True)

    # ─── Chat Replay with Sentiment ─────────────────────────────────────
    st.subheader("💬 Chat Replay with Sentiment Signals")
    st.markdown(
        "🟢 Green = safe to cross-sell &nbsp;&nbsp; "
        "🟡 Yellow = wait &nbsp;&nbsp; "
        "🔴 Red = hold back"
    )
    render_chat_with_sentiment(results_df)

    # ─── Cross-sell Effectiveness (if data has outcome labels) ──────────
    if 'category' in results_df.columns:
        sell_attempts = results_df[results_df['category'].str.contains('cross_sell_attempt', na=False)]
        sell_success = results_df[results_df['category'].str.contains('cross_sell_success', na=False)]
        sell_failure = results_df[results_df['category'].str.contains('cross_sell_failure', na=False)]
        
        if len(sell_attempts) > 0:
            st.markdown("---")
            st.subheader("📊 Cross-sell Outcome Analysis")
            
            # Get the sentiment BEFORE the cross-sell attempt
            for _, attempt in sell_attempts.iterrows():
                msg_num = attempt['message_number']
                prev_customer = results_df[
                    (results_df['message_number'] < msg_num) & 
                    (results_df['speaker'] == 'customer')
                ]
                if not prev_customer.empty:
                    pre_sell_sentiment = prev_customer.iloc[-1]['sentiment_score']
                    conv_id = attempt['conversation_id']
                    
                    # Check outcome
                    outcome_msgs = results_df[
                        (results_df['conversation_id'] == conv_id) & 
                        (results_df['message_number'] > msg_num)
                    ]
                    success = any(outcome_msgs['category'].str.contains('cross_sell_success', na=False))
                    
                    emoji = "✅" if success else "❌"
                    st.markdown(
                        f"**{conv_id}:** Agent attempted cross-sell at sentiment "
                        f"**{pre_sell_sentiment:+.2f}** → {emoji} "
                        f"{'Accepted' if success else 'Rejected'}"
                    )

    # ─── All Conversations Overview ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 All Conversations Overview")
    
    with st.spinner("Analyzing all conversations..."):
        all_results = analyze_conversation(df, analyzer, sell_threshold, hold_threshold)
    
    # Summary by conversation
    summary = (
        all_results[all_results['speaker'] == 'customer']
        .groupby('conversation_id')
        .agg(
            brand=('brand', 'first'),
            category=('category', 'first'),
            avg_sentiment=('sentiment_score', 'mean'),
            final_sentiment=('sentiment_score', 'last'),
            messages=('message_number', 'count'),
            sell_windows=('crosssell_decision', lambda x: sum('CROSS-SELL' in str(v) for v in x)),
        )
        .round(3)
        .sort_values('avg_sentiment', ascending=False)
    )
    
    st.dataframe(summary, use_container_width=True)
    
    # Sentiment distribution by brand
    if 'brand' in all_results.columns:
        brand_sentiment = (
            all_results[all_results['speaker'] == 'customer']
            .groupby('brand')['sentiment_score']
            .mean()
            .reset_index()
        )
        fig_brand = px.bar(
            brand_sentiment, x='brand', y='sentiment_score',
            title="Average Customer Sentiment by Brand",
            color='sentiment_score',
            color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
            range_color=[-1, 1]
        )
        fig_brand.update_layout(height=350)
        st.plotly_chart(fig_brand, use_container_width=True)

    # ─── Cross-Sell Performance Dashboard ───────────────────────────────
    if 'category' in all_results.columns:
        st.markdown("---")
        st.subheader("📊 Cross-Sell Performance by Sentiment Zone")
        st.markdown(
            "**The business case:** Does cross-selling at the right sentiment "
            "moment actually convert better? This table answers that question."
        )

        # Classify every customer message into a zone
        cust_all = all_results[all_results['speaker'] == 'customer'].copy()
        
        def get_zone(score):
            if score >= sell_threshold:
                return "🟢 Green (Cross-sell)"
            elif score <= hold_threshold:
                return "🔴 Red (Hold back)"
            else:
                return "🟡 Yellow (Wait)"
        
        cust_all['zone'] = cust_all['sentiment_score'].apply(get_zone)

        # Count total messages per zone
        zone_msg_counts = cust_all['zone'].value_counts().to_dict()

        # Now analyze cross-sell attempts and their outcomes per zone
        # For each cross-sell attempt, find the customer sentiment BEFORE it
        attempts_data = []
        for conv_id in all_results['conversation_id'].unique():
            conv = all_results[all_results['conversation_id'] == conv_id]
            sell_attempts_conv = conv[
                conv['category'].str.contains('cross_sell_attempt', na=False)
            ]
            for _, attempt in sell_attempts_conv.iterrows():
                msg_num = attempt['message_number']
                # Get last customer message before this attempt
                prev_cust = conv[
                    (conv['message_number'] < msg_num) & 
                    (conv['speaker'] == 'customer')
                ]
                if prev_cust.empty:
                    continue
                
                pre_score = prev_cust.iloc[-1]['sentiment_score']
                zone = get_zone(pre_score)
                
                # Check if next customer message was a success
                next_msgs = conv[conv['message_number'] > msg_num]
                success = any(
                    next_msgs['category'].str.contains('cross_sell_success', na=False)
                )
                
                attempts_data.append({
                    'conversation_id': conv_id,
                    'zone': zone,
                    'pre_sell_sentiment': pre_score,
                    'success': success,
                    'brand': attempt.get('brand', 'Unknown')
                })

        if attempts_data:
            attempts_df = pd.DataFrame(attempts_data)
            
            # Build the summary table
            zone_order = [
                "🟢 Green (Cross-sell)", 
                "🟡 Yellow (Wait)", 
                "🔴 Red (Hold back)"
            ]
            
            dashboard_rows = []
            for zone in zone_order:
                total_msgs = zone_msg_counts.get(zone, 0)
                zone_attempts = attempts_df[attempts_df['zone'] == zone]
                num_attempts = len(zone_attempts)
                num_success = zone_attempts['success'].sum() if num_attempts > 0 else 0
                num_failed = num_attempts - num_success
                success_rate = (num_success / num_attempts * 100) if num_attempts > 0 else 0
                
                dashboard_rows.append({
                    'Sentiment Zone': zone,
                    'Customer Messages': total_msgs,
                    'Cross-sell Attempts': num_attempts,
                    'Accepted ✅': int(num_success),
                    'Rejected ❌': int(num_failed),
                    'Success Rate': f"{success_rate:.0f}%"
                })
            
            dashboard_df = pd.DataFrame(dashboard_rows)
            st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

            # Visual: success rate by zone
            chart_data = []
            for zone in zone_order:
                zone_attempts = attempts_df[attempts_df['zone'] == zone]
                if len(zone_attempts) > 0:
                    rate = zone_attempts['success'].mean() * 100
                else:
                    rate = 0
                chart_data.append({'Zone': zone, 'Success Rate (%)': rate})
            
            chart_df = pd.DataFrame(chart_data)
            
            fig_perf = go.Figure()
            colors = ['#28a745', '#ffc107', '#dc3545']
            for i, row in chart_df.iterrows():
                fig_perf.add_trace(go.Bar(
                    x=[row['Zone']],
                    y=[row['Success Rate (%)']],
                    marker_color=colors[i],
                    name=row['Zone'],
                    text=f"{row['Success Rate (%)']:.0f}%",
                    textposition='outside'
                ))
            
            fig_perf.update_layout(
                title="Cross-Sell Success Rate by Sentiment Zone",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[0, 110]),
                height=400,
                showlegend=False,
                template="plotly_white"
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            # Key insight callout
            green_attempts = attempts_df[attempts_df['zone'] == "🟢 Green (Cross-sell)"]
            red_attempts = attempts_df[attempts_df['zone'] == "🔴 Red (Hold back)"]
            green_rate = green_attempts['success'].mean() * 100 if len(green_attempts) > 0 else 0
            red_rate = red_attempts['success'].mean() * 100 if len(red_attempts) > 0 else 0
            
            if green_rate > red_rate and len(green_attempts) > 0:
                multiplier = green_rate / red_rate if red_rate > 0 else float('inf')
                if multiplier == float('inf'):
                    insight = (
                        f"Cross-selling in the green zone had a **{green_rate:.0f}% success rate** "
                        f"while red zone attempts had **0% success**. "
                        f"Sentiment-aware timing eliminates wasted pitches entirely."
                    )
                else:
                    insight = (
                        f"Cross-selling in the green zone converts at "
                        f"**{multiplier:.1f}x the rate** of the red zone "
                        f"({green_rate:.0f}% vs {red_rate:.0f}%). "
                        f"This validates using sentiment thresholds to time cross-sell attempts."
                    )
                st.success(f"💡 **Key Insight:** {insight}")
            
            # Detailed attempt log
            with st.expander("📋 Detailed Cross-Sell Attempt Log"):
                for _, row in attempts_df.iterrows():
                    emoji = "✅" if row['success'] else "❌"
                    st.markdown(
                        f"**{row['conversation_id']}** ({row['brand']}) — "
                        f"Pitched at sentiment **{row['pre_sell_sentiment']:+.2f}** "
                        f"[{row['zone']}] → {emoji} "
                        f"{'Accepted' if row['success'] else 'Rejected'}"
                    )
        else:
            st.info("No cross-sell attempts found in the dataset to analyze.")

    # ─── Try Your Own Message ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧪 Try It: Analyze Any Message")
    
    test_message = st.text_area(
        "Paste a customer message to analyze",
        placeholder="e.g., 'I've been waiting 2 weeks and nobody responds to my emails. This is terrible service.'",
        height=80
    )
    
    if test_message:
        score = get_sentiment_score(test_message, analyzer)
        decision, confidence, reasoning = classify_crosssell_window(
            score, sell_threshold, hold_threshold
        )
        
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Sentiment Score", f"{score:+.3f}")
        tc2.metric("Decision", decision)
        tc3.metric("Confidence", f"{confidence:.0f}%")
        st.info(f"💡 **Reasoning:** {reasoning}")


if __name__ == "__main__":
    main()
