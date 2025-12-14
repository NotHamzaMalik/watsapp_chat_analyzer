import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np
import calendar
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud
import emoji
from nltk.corpus import stopwords
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

sia = SentimentIntensityAnalyzer()
extractor = URLExtract()
STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['media', 'omitted', 'message','deleted','image', 'video', 'audio', 'sticker', 'gif', 'document'])

try:
    with open('stop words.txt', 'r', encoding='utf-8') as f:
        custom_stopwords = set(line.strip().lower() for line in f)
    STOP_WORDS = list(set(STOP_WORDS) | custom_stopwords)
except:
    pass

def split_sentences(text):
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    return [s.strip() for s in sentences if s.strip()]

def mmr_select(doc_embedding, candidate_embeddings, candidates, top_n=5, lambda_param=0.7):
    if len(candidates) == 0:
        return []

    selected = []
    selected_idx = []

    sim_to_doc = cosine_similarity(candidate_embeddings, doc_embedding.reshape(1, -1)).ravel()
    sim_between = cosine_similarity(candidate_embeddings)

    first = np.argmax(sim_to_doc)
    selected.append(candidates[first])
    selected_idx.append(first)

    while len(selected) < top_n and len(selected_idx) < len(candidates):
        mmr_scores = []
        for i in range(len(candidates)):
            if i in selected_idx:
                mmr_scores.append((-1e9, i))
                continue
            
            relevance = sim_to_doc[i]
            redundancy = max(sim_between[i][j] for j in selected_idx)
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((score, i))

        mmr_scores.sort(reverse=True)
        chosen = mmr_scores[0][1]
        selected.append(candidates[chosen])
        selected_idx.append(chosen)

    return selected

def generate_summary(df, user, n_sentences=5):
    if user == "Overall":
        temp_df = df
    else:
        temp_df = df[df["user"] == user]
    
    clean_msgs = temp_df[~temp_df['message'].str.contains('<Media omitted>', case=False, na=False)]
    text = " ".join(clean_msgs['message'].astype(str).tolist())
    
    if len(text.strip()) < 50:
        return "Not enough text data to generate a summary."
    
    sentences = split_sentences(text)
    if len(sentences) < 3:
        return "Need more sentences for summarization."
    
    if len(sentences) > 5000:
        sentences = sentences[-5000:]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(sentences)
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            return "Could not process text for summarization."
        
        doc_embedding = X.mean(axis=0)
        sentence_vectors = X.toarray()
        
        if len(sentences) <= n_sentences:
            return "\n\n".join([f"• {s}" for s in sentences])
        
        sims = cosine_similarity(sentence_vectors, np.asarray(doc_embedding))[:, 0]
        ranked_idx = np.argsort(-sims)
        
        top_k = n_sentences
        top_candidates_idx = ranked_idx[:min(len(ranked_idx), top_k * 4)]
        
        candidates = [sentences[i] for i in top_candidates_idx]
        candidate_vectors = sentence_vectors[top_candidates_idx]
        
        selected = mmr_select(
            np.asarray(doc_embedding).ravel(),
            candidate_vectors,
            candidates,
            top_n=top_k,
            lambda_param=0.7
        )
        
        if not selected:
            selected = [sentences[i] for i in ranked_idx[:top_k]]
        
        return "\n\n".join([f"• {s}" for s in selected])
    except Exception as e:
        return f"Could not generate summary. Error: {str(e)}"

def fetch_stats(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    messages = temp_df["message"].astype(str)
    num_msgs = len(messages)
    word_count = sum(len(m.split()) for m in messages)
    media_count = messages.str.contains(r'<Media omitted>', na=False, regex=True).sum()
    link_count = messages.apply(lambda m: len(extractor.find_urls(m))).sum()
    return num_msgs, word_count, media_count, link_count

def most_busy_users(df):
    return df["user"].value_counts().head(10)

def week_activity_map(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    return temp_df['day_name'].value_counts()

def month_activity_map(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    return temp_df['month'].value_counts()

def year_activity_map(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    return temp_df['year'].value_counts().sort_index()

def daily_activity_map(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    return temp_df.groupby('only_date').count()['message']

def activity_heatmap(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    pivot = temp_df.pivot_table(index="day_name", columns="hour", values="message", aggfunc="count").fillna(0)
    pivot = pivot.reindex(list(calendar.day_name))
    pivot = pivot.reindex(columns=range(24), fill_value=0)
    return pivot

def create_wordcloud(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    temp = temp_df[~temp_df["message"].str.contains(r'<Media omitted>', na=False, regex=True)]
    text = " ".join(temp["message"].astype(str).str.lower())
    
    if not text.strip():
        return None

    wc = WordCloud(width=800, height=400, background_color="white", 
                   stopwords=STOP_WORDS, min_font_size=10,
                   colormap='viridis').generate(text)
    return wc

def most_common_words(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    words = []
    for msg in temp_df["message"].astype(str):
        if '<Media omitted>' in msg: continue
        
        for w in msg.lower().split():
            cleaned_w = re.sub(r'[^a-zA-Z0-9]', '', w)
            if cleaned_w and cleaned_w not in STOP_WORDS and len(cleaned_w) > 2:
                words.append(cleaned_w)
    return pd.DataFrame(Counter(words).most_common(20), columns=["word", "count"])

def emoji_analysis(user, df):
    temp_df = df if user == "Overall" else df[df["user"] == user]
    emojis_list = []
    for msg in temp_df["message"].astype(str):
        emojis_list.extend([c for c in msg if c in emoji.EMOJI_DATA])
    return pd.DataFrame(Counter(emojis_list).most_common(10), columns=["emoji", "count"])

def analyze_emotional_peaks(df):
    daily_sentiment = df.groupby('only_date')['sentiment'].mean()
    if daily_sentiment.empty:
        return None, 0, None, 0
    peak_pos_date = daily_sentiment.idxmax()
    peak_pos_score = daily_sentiment.max()
    peak_neg_date = daily_sentiment.idxmin()
    peak_neg_score = daily_sentiment.min()
    return peak_pos_date, peak_pos_score, peak_neg_date, peak_neg_score

def analyze_response_time(df):
    df_resp = df.sort_values('date').copy()
    df_resp['time_diff'] = df_resp['date'].diff().dt.total_seconds()
    df_resp['user_prev'] = df_resp['user'].shift(1)
    response_df = df_resp[df_resp['user'] != df_resp['user_prev']].dropna(subset=['time_diff'])
    response_rank = response_df.groupby('user')['time_diff'].mean().sort_values()
    return response_rank

def apply_filters(df):
    import streamlit as st
    st.sidebar.title("Configuration")
    user_list = ['Overall'] + sorted(list(df['user'].unique()))
    
    multi_user_mode = st.sidebar.checkbox("Enable Multi-User Comparison")
    
    if multi_user_mode:
        selected_users = st.sidebar.multiselect("Select Users for Comparison", user_list[1:], default=user_list[1:min(3, len(user_list))])
        selected_user = "Comparison"
    else:
        selected_users = [st.sidebar.selectbox("Select User for Analysis", user_list)]
        selected_user = selected_users[0]

    temp_df = df.copy()
    if not multi_user_mode and selected_user != 'Overall':
        temp_df = temp_df[temp_df['user'] == selected_user]

    if not temp_df.empty:
        min_date_raw = temp_df['only_date'].min()
        max_date_raw = temp_df['only_date'].max()
        start_date = st.sidebar.date_input("Start Date", min_date_raw, min_value=min_date_raw, max_value=max_date_raw)
        end_date = st.sidebar.date_input("End Date", max_date_raw, min_value=min_date_raw, max_value=max_date_raw)
        temp_df = temp_df[(temp_df['only_date'] >= start_date) & (temp_df['only_date'] <= end_date)]

    return temp_df, selected_user, selected_users, multi_user_mode

def plot_sentiment_timeline(df):
    import plotly.graph_objects as go
    import streamlit as st

    if df.empty:
        st.info("No data available for sentiment analysis.")
        return go.Figure()

    timeline = df.set_index('date')['sentiment'].resample('h').mean().fillna(method='ffill')
    if timeline.empty:
        st.info("No sentiment data after resampling.")
        return go.Figure()

    timeline_smooth = timeline.rolling(window=24, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline_smooth.index,
        y=timeline_smooth.values,
        mode='lines',
        name='24H Rolling Avg',
        line=dict(color='#636EFA', width=3),
        fill='tozeroy'
    ))
    fig.add_trace(go.Scatter(
        x=timeline.index,
        y=timeline.values,
        mode='markers',
        name='Hourly Sentiment',
        marker=dict(size=4, color=timeline.values, colorscale='RdYlGn'),
        opacity=0.6
    ))
    fig.update_layout(
        title='📈 Sentiment Timeline',
        template='plotly_dark',
        height=400
    )

    return fig

def plot_multi_user_comparison(df, selected_users):
    if len(selected_users) < 2: return go.Figure()
    fig = go.Figure()
    colors = px.colors.qualitative.Set3
    for idx, user in enumerate(selected_users):
        user_df = df[df['user'] == user]
        if not user_df.empty:
            user_timeline = user_df.set_index('date')['sentiment'].resample('D').mean().fillna(method='ffill')
            fig.add_trace(go.Scatter(x=user_timeline.index, y=user_timeline.values, mode='lines+markers', name=user, line=dict(color=colors[idx % len(colors)], width=2)))
    fig.update_layout(title='Multi-User Sentiment Comparison', template='plotly_white', height=400)
    return fig

def plot_communication_network(df):
    if df.empty or len(df['user'].unique()) < 2:
        return go.Figure()
    df_sorted = df.sort_values('date')
    df_sorted['next_user'] = df_sorted['user'].shift(-1)
    df_sorted['time_diff'] = df_sorted['date'].shift(-1) - df_sorted['date']
    edges = df_sorted[df_sorted['time_diff'] < pd.Timedelta(minutes=5)]
    edges = edges[edges['user'] != edges['next_user']]
    if edges.empty:
        return go.Figure()
    edge_counts = edges.groupby(['user', 'next_user']).size().reset_index(name='count')
    edge_counts = edge_counts[edge_counts['count'] > 1]
    all_users = list(set(edge_counts['user'].tolist() + edge_counts['next_user'].tolist()))
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_users,
            color=px.colors.qualitative.Set3[:len(all_users)]
        ),
        link=dict(
            source=[all_users.index(src) for src in edge_counts['user']],
            target=[all_users.index(tgt) for tgt in edge_counts['next_user']],
            value=edge_counts['count'],
            color='rgba(99, 110, 250, 0.4)'
        )
    )])
    fig.update_layout(
        title_text="Communication Flow Network",
        font_size=15,
        height=700
    )
    return fig

def plot_message_length_distribution(df):
    if df.empty: return go.Figure()
    df['message_length'] = df['message'].astype(str).apply(len)
    fig = px.box(df, x='user', y='message_length', color='user', points="all", template='plotly_white')
    fig.update_layout(title='Message Length Distribution', showlegend=False, height=400)
    return fig



def display_persona_generator(df, selected_user):
    import streamlit as st
    import matplotlib.pyplot as plt
    import plotly.express as px

    st.subheader(f"Language Style Analysis: {selected_user}")

    # Word Cloud
    st.markdown("#### 🖌️ Word Cloud")
    wc = create_wordcloud(selected_user, df)
    if wc:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Not enough text data for Word Cloud.")

    st.markdown("---")

    # Top Words
    st.markdown("#### 📊 Top Words")
    mcw = most_common_words(selected_user, df)
    if not mcw.empty:
        fig_words = px.bar(
            mcw.head(10),
            x='count',
            y='word',
            orientation='h',
            template='plotly_white',
            color='count'
        )
        fig_words.update_layout(yaxis={'categoryorder': 'total ascending'}, 
                                xaxis_title=None, yaxis_title=None, height=300)
        st.plotly_chart(fig_words, use_container_width=True)
    else:
        st.info("No significant words found")

    st.markdown("---")

    # Emoji Usage
    st.markdown("#### 😀 Emoji Usage")
    emoji_df = emoji_analysis(selected_user, df)
    if not emoji_df.empty:
        fig_emoji = px.pie(
            emoji_df,
            values='count',
            names='emoji',
            hole=0.4,
            template='plotly_white'
        )
        st.plotly_chart(fig_emoji, use_container_width=True)
    else:
        st.info("No emojis found")

def show_top_links_and_media(df):
    import streamlit as st
    st.subheader("Shared Content Analysis")

    # Most Shared Links
    st.markdown("#### 🔗 Most Shared Links")
    all_links = []
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    for msg in df["message"].astype(str):
        all_links.extend(re.findall(url_pattern, msg))
        
    if all_links:
        link_df = pd.DataFrame(Counter(all_links).most_common(10), columns=['URL', 'Count'])
        fig_links = px.bar(
            link_df.head(8),
            x='Count',
            y='URL',
            orientation='h',
            template='plotly_white'
        )
        fig_links.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300)
        st.plotly_chart(fig_links, use_container_width=True)
    else:
        st.info("No links shared")

    st.markdown("---")
    # Media & File Sharing
    st.markdown("#### 📁 Media & File Sharing")
    media_count = df["message"].str.contains(r'<Media omitted>', na=False, regex=True).sum()
    fig_media = go.Figure(go.Indicator(
        mode="gauge+number",
        value=media_count,
        title={'text': "Media Files"},
        gauge={
            'axis': {'range': [0, max(media_count*2, 100)]},
            'bar': {'color': "darkblue"}
        }
    ))
    fig_media.update_layout(height=300)
    st.plotly_chart(fig_media, use_container_width=True)


def generate_activity_report(df, selected_user):
    import streamlit as st
    st.subheader("Activity Pattern Analysis")
    
    st.markdown("#### 📅 Monthly Timeline")
    monthly_data = month_activity_map(selected_user, df)
    month_order = list(calendar.month_name)[1:]
    monthly_data = monthly_data.reindex(month_order, fill_value=0)
    
    monthly_df = pd.DataFrame({'Month': monthly_data.index, 'Messages': monthly_data.values})
    fig_month = px.bar(monthly_df, x='Month', y='Messages', template='plotly_dark', color='Messages', color_continuous_scale='Viridis')
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("#### 📈 Daily Activity Trends")
    daily_data = daily_activity_map(selected_user, df)
    daily_df = pd.DataFrame({'Date': daily_data.index, 'Messages': daily_data.values})
    fig_daily_trend = px.line(daily_df, x='Date', y='Messages', template='plotly_dark')
    st.plotly_chart(fig_daily_trend, use_container_width=True)

    col1, col2 = st.columns(2)
    
    st.markdown("#### 🗓️ Weekly Pattern")
    weekly_data = week_activity_map(selected_user, df)
    day_order = list(calendar.day_name)
    weekly_data = weekly_data.reindex(day_order, fill_value=0)
    weekly_df = pd.DataFrame({'Day': weekly_data.index, 'Messages': weekly_data.values})
    fig_week = px.bar(weekly_df, x='Day', y='Messages', template='plotly_dark', color='Messages')
    st.plotly_chart(fig_week, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 🗓️ Yearly Pattern")
    yearly_data = year_activity_map(selected_user, df)
    yearly_df = pd.DataFrame({'Year': yearly_data.index.astype(str), 'Messages': yearly_data.values})
    fig_year = px.bar(yearly_df, x='Year', y='Messages', template='plotly_dark', color='Messages')
    st.plotly_chart(fig_year, use_container_width=True)

    
    st.markdown("#### 🔥 Time Heatmap")
    heatmap_data = activity_heatmap(selected_user, df)
    fig_heatmap = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index, colorscale='Viridis'))
    fig_heatmap.update_layout(xaxis_title="Hour (24h)", height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)