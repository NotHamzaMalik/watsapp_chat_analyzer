# 💬 WhatsApp Chat Analyzer 

**WhatsApp Chat Analyzer ** is a powerful, interactive analytics dashboard built with **Python** and **Streamlit** for deep exploration, visualization, and intelligent summarization of WhatsApp chat histories.

It transforms raw WhatsApp exports into meaningful insights using **Natural Language Processing (NLP)**, **sentiment analysis**, and **advanced visualizations**.

---

## ✨ Key Features

### 🤖 Intelligent Chat Summarization
- Extractive summarization using **TF-IDF + Maximal Marginal Relevance (MMR)**
- Produces **concise, non-redundant, and context-aware summaries**
- Supports **per-user** and **overall conversation** summaries
- Fully offline & fast (no API calls)

---

### 😊 Sentiment Analysis
- Message-level sentiment scoring using **VADER (NLTK)**
- Hourly & daily sentiment timelines
- Detects:
  - 📈 Most positive day
  - 📉 Most negative day
- Multi-user sentiment comparison

---

### 📊 Activity & Time Analysis
- Message frequency analysis:
  - Daily
  - Weekly
  - Monthly
  - Yearly
- **Time Heatmap** (Day × Hour)
- Identifies peak activity hours & days

---

### 👤 User Behavior Profiling
- Most active users
- Word clouds (persona analysis)
- Top cleaned words (stopword filtered)
- Emoji usage breakdown
- Media & link sharing behavior

---

### 🔁 Communication Flow Analysis
- **Sankey diagram** to visualize turn-taking & reply patterns
- Highlights conversational dominance and interaction paths

---

### 🧭 Interactive Dashboard
- User-based filtering
- Date range filtering
- Multi-user comparison mode
- Clean, responsive UI with Plotly visuals

---

## ⚙️ Tech Stack

| Layer | Tools |
|-----|------|
| Framework | **Streamlit** |
| Data Processing | **Pandas, NumPy** |
| NLP | **NLTK (VADER, Tokenization)** |
| Summarization | **TF-IDF, Cosine Similarity, MMR** |
| Visualization | **Plotly, Matplotlib, WordCloud** |
| Utilities | **URLExtract, Emoji** |

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python **3.8 or higher**
- pip (Python package manager)

---

### 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/whatsapp-chat-analyzer-pro.git
cd whatsapp-chat-analyzer-pro

Install dependencies
pip install -r requirements.txt


▶️ Run the Application
streamlit run app.py

📁 Data Preparation
Export WhatsApp Chat
Open WhatsApp on your phone
Go to the target chat (group or individual)
Tap More (⋮) → Export Chat
Select Without Media (recommended)
Upload the .txt file or .zip containing it


🗂️ Project Structure
whatsapp-chat-analyzer-pro/
│
├── app.py                # Streamlit UI & controller
├── helper.py             # Core analytics, NLP & visualization logic
├── preprocessing.py      # Chat parsing & feature engineering
├── requirements.txt      # Python dependencies
├── stop words.txt        # Custom stopword list (optional)
└── README.md             # Project documentation


🧠 Architecture Overview

preprocessing.py
Parses raw WhatsApp text
Converts to structured DataFrame
Extracts date/time features
Computes initial sentiment scores
helper.py
Statistical analysis
Activity maps & heatmaps
NLP pipelines
TF-IDF + MMR summarizer
Plotly visualizations
app.py
UI layout & navigation
Sidebar filters
Tab orchestration
State management



🙌 Author

Hamza Malik
AI & ML Student 
📍 Islamabad