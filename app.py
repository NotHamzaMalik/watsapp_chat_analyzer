import streamlit as st
import pandas as pd
import zipfile
import io
from preprocessing import preprocess
import helper
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def main():
    st.set_page_config(layout="wide", page_title="Watsapp Chat Analyzer", page_icon="💬")
    
    st.title("💬 WhatsApp Chat Analyzer")
    st.markdown("Analyze your chat history with advanced sentiment analysis, summarization, and visualization tools.")
    
    # --- Session State Initialization ---
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False

    uploaded_file = st.file_uploader(
        "Upload Chat File", 
        type=["txt", "zip", "csv"], 
        help="Supports .txt, .zip, or .csv formats."
    )
    
    # Reset analysis if file is removed or changed
    if uploaded_file is None:
        st.session_state['analyzed'] = False
    
    df = pd.DataFrame()

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        data = None
        
        # We only process the file if we haven't already, or if we need to reload it.
        # Ideally, we cache this, but for now, we process normally.
        with st.spinner(f'Processing {file_ext.upper()} file...'):
            try:
                if file_ext == "txt":
                    bytes_data = uploaded_file.getvalue()
                    data = bytes_data.decode("utf-8")
                
                elif file_ext == "zip":
                    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
                        txt_files = [name for name in zf.namelist() if name.endswith('.txt')]
                        if not txt_files:
                            st.error("Error: ZIP file does not contain a .txt chat file.")
                            return
                        with zf.open(txt_files[0]) as chat_file:
                            data = chat_file.read().decode("utf-8")

                elif file_ext == "csv":
                    df = pd.read_csv(uploaded_file)
                    if not all(col in df.columns for col in ['date', 'user', 'message']):
                         st.error("CSV must contain 'date', 'user', 'message' columns")
                         return
                    df['date'] = pd.to_datetime(df['date'])
                    df['only_date'] = df['date'].dt.date
                    if 'sentiment' not in df.columns:
                        from nltk.sentiment.vader import SentimentIntensityAnalyzer
                        sia = SentimentIntensityAnalyzer()
                        df['sentiment'] = df['message'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])
                    df = df[df['user'] != 'group_notification'].reset_index(drop=True)

                if data is not None:
                    df = preprocess(data)
                
                if isinstance(df, tuple):
                    st.error("Parsing Error: Format mismatch.")
                    st.stop()
                
                if df.empty:
                    st.error("Error: File is empty or malformed.")
                    return

            except Exception as e:
                st.error(f"Processing error: {e}")
                return
        
        st.success("File Processed Successfully!")
        
        # Apply filters (Sidebar)
        filtered_df, selected_user, selected_users, multi_user_mode = helper.apply_filters(df.copy())
        
        st.sidebar.markdown("---")
        
        # --- FIXED LOGIC HERE ---
        # Instead of just checking if the button is clicked NOW, 
        # we check if it was clicked OR if we are already in analysis mode.
        analyze_clicked = st.sidebar.button("📊 Analyze Chat", type="primary", use_container_width=True)
        
        if analyze_clicked:
            st.session_state['analyzed'] = True
            
        # Only show analysis if the state is True
        if st.session_state['analyzed']:
            
            st.markdown("---")
            if len(filtered_df) < 5:
                st.warning("Not enough data points selected.")
                return

            # Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Overview", 
                "Sentiment Analysis", 
                "User Behavior", 
                "Advanced Insights",
                "Text Summarization"
            ])
            
            # --- TAB 1: OVERVIEW ---
            with tab1:
                st.subheader("Conversation Metrics Dashboard")
                if multi_user_mode:
                    cols = st.columns(min(4, len(selected_users)))
                    for idx, user in enumerate(selected_users):
                        with cols[idx % len(cols)]:
                            user_df = filtered_df[filtered_df['user'] == user]
                            num_msgs, words, media, links = helper.fetch_stats(user, user_df)
                            st.metric(f"{user}", f"{num_msgs:,}", delta=f"{words:,} words", delta_color="off")
                else:
                    num_msgs, words, media, links = helper.fetch_stats(selected_user, filtered_df)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Messages", f"{num_msgs:,}")
                    c2.metric("Words", f"{words:,}")
                    c3.metric("Media", f"{media:,}")
                    c4.metric("Links", f"{links:,}")

                st.markdown("---")
                if selected_user == 'Overall' or multi_user_mode:
                    st.subheader("User Activity Comparison")
                    compare_users = selected_users if multi_user_mode else helper.most_busy_users(filtered_df).head(5).index.tolist()
                    comparison_data = []
                    for user in compare_users:
                        user_df = filtered_df[filtered_df['user'] == user]
                        msgs, words, media, links = helper.fetch_stats(user, user_df)
                        comparison_data.append({'User': user, 'Messages': msgs, 'Avg Words per Msg': words/msgs if msgs > 0 else 0})
                    comp_df = pd.DataFrame(comparison_data)
                    fig = px.bar(comp_df, x='User', y=['Messages', 'Avg Words per Msg'], barmode='group', template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: SENTIMENT ---
            with tab2:
                st.subheader("Advanced Sentiment Analysis")

            # Sentiment Timeline
                st.markdown("#### 📈 Sentiment Timeline")
                st.plotly_chart(helper.plot_sentiment_timeline(filtered_df), use_container_width=True)

                st.markdown("---")

                # Multi-User Comparison or Sentiment Distribution
                st.markdown("#### 📊 Sentiment Distribution / Comparison")
                if multi_user_mode:
                    st.plotly_chart(helper.plot_multi_user_comparison(filtered_df, selected_users), use_container_width=True)
                else:
                    sentiment_dist = filtered_df['sentiment_category'].value_counts()
                    fig = px.pie(
                        values=sentiment_dist.values,
                        names=sentiment_dist.index,
                        hole=0.5,
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Emotional Peaks
                peak_pos_date, peak_pos_score, peak_neg_date, peak_neg_score = helper.analyze_emotional_peaks(filtered_df)
                st.markdown("#### 📌 Emotional Peaks")
                if peak_pos_date:
                    st.success(f"📈 Peak Positivity: {peak_pos_date.strftime('%b %d, %Y')}")
                if peak_neg_date:
                    st.error(f"📉 Peak Negativity: {peak_neg_date.strftime('%b %d, %Y')}")

            # --- TAB 3: USER BEHAVIOR ---
            with tab3:
                helper.generate_activity_report(filtered_df, selected_user)
                st.markdown("---")
                if not multi_user_mode:
                    helper.display_persona_generator(filtered_df, selected_user)
                st.markdown("---")
                helper.show_top_links_and_media(filtered_df)
                


            # --- TAB 4: ADVANCED INSIGHTS ---
            with tab4:
                st.subheader("📡 Communication Network")
                st.plotly_chart(
                helper.plot_communication_network(filtered_df),
                use_container_width=True
                )
                st.markdown("---")    

                st.subheader("📊 Message Length Distribution")
                st.plotly_chart(
                helper.plot_message_length_distribution(filtered_df),
                use_container_width=True
                )
                st.markdown("---")

                response_rank = None



               

                if len(filtered_df['user'].unique()) > 1:
                    response_rank = helper.analyze_response_time(filtered_df)

                if response_rank is not None and not response_rank.empty:
                    response_df = pd.DataFrame({
                        'User': response_rank.index,
                        'Avg Response Time (min)': response_rank.values / 60
                    })

                    fig = px.bar(
                        response_df,
                        x='User',
                        y='Avg Response Time (min)',
                        color='Avg Response Time (min)',
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)



            # --- TAB 5: TEXT SUMMARIZATION (FIXED) ---
            with tab5:
                st.subheader("🤖 Smart Conversation Summarizer")
                st.info("Uses TF-IDF & MMR to extract the most relevant and non-redundant sentences from the chat.")

                col_sum_1, col_sum_2 = st.columns([1, 3])
                
                with col_sum_1:
                    # User Selection for Summary
                    unique_users = ['Overall'] + list(filtered_df['user'].unique())
                    sum_target_user = st.selectbox("Select User/Target to Summarize", unique_users)
                    
                    # Summary Length Slider
                    summary_sentences = st.slider("Summary Length (Sentences)", min_value=3, max_value=20, value=5)
                    
                    generate_btn = st.button("Generate Summary", type="primary")

                with col_sum_2:
                    if generate_btn:
                        with st.spinner("Analyzing text and generating summary... (This uses TF-IDF & MMR)"):
                            summary_result = helper.generate_summary(filtered_df, sum_target_user, n_sentences=summary_sentences)
                            
                            st.markdown(f"### 📝 Summary for: {sum_target_user}")
                            if "Error" in summary_result or "Not enough" in summary_result:
                                st.warning(summary_result)
                            else:
                                st.success("Summary Generated Successfully")
                                st.info(summary_result)

            # Sidebar Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "✨ Made By **Hamza Malik**\n\n"
        "🔗 [GitHub](https://github.com/NotHamzaMalik) | "
        "📧 [Contact](mailto:glactic@gmail.com)"
    )
    st.sidebar.markdown("© 2025 WhatsApp Chat Analyzer")




    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey;'>Made by Hamza Malik</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()