import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import emoji # Make sure to install: pip install emoji
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer # For sentiment
import re
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist
from langdetect import detect

# Download NLTK resources if not already downloaded (run this once locally)
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('punkt') # For tokenization

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Vignesh & Sukyoung's Chat Story")

# --- Global Variables / Constants ---
AUTHOR_1 = "Vikke"
AUTHOR_2 = "Sukyoung"
MEDIA_MESSAGE = "<Media omitted>"
LOVE_KEYWORDS = [
    'love', 'miss u', 'miss you', 'luv u', 'luv you', 'ily', 'i love you', 'my love',
    'heart', 'hearts', ' babe', 'baby', 'darling', 'sweetheart', 'honey',
    '❤️', '😘', '😍', '🥰', '💕', '💖', '💗', '💓', '💞', '💘', '💌'
] # Add your own specific terms!

# --- Helper Functions ---

@st.cache_data # Cache the data loading and preprocessing
def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None, names=[
            'DateTime_Str', 'Author', 'Message', 'Date_Only_Str',
            'Time_Only_Str', 'Hour_Parsed', 'Day_Name_Parsed'
        ])
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.error("Please ensure your CSV has 7 columns and is comma-separated with no header row.")
        return None

    # Convert to datetime
    try:
        df['Timestamp'] = pd.to_datetime(df['DateTime_Str'])
    except Exception as e:
        st.error(f"Error parsing DateTime_Str column: {e}")
        st.info("Expected format: YYYY-MM-DD HH:MM:SS")
        return None

    # Basic Cleaning
    df.dropna(subset=['Message', 'Author'], inplace=True) # Drop rows where message or author is NaN
    df['Author'] = df['Author'].str.strip()
    df['Message'] = df['Message'].str.strip()

    # Filter out potential system messages or rows with unexpected authors if any
    # For this project, we assume only AUTHOR_1 and AUTHOR_2 are relevant
    df = df[df['Author'].isin([AUTHOR_1, AUTHOR_2])]
    if df.empty:
        st.warning(f"No messages found for authors '{AUTHOR_1}' or '{AUTHOR_2}'. Please check author names in your CSV.")
        return None

    # Feature Engineering
    df['Date_Only'] = df['Timestamp'].dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Num'] = df['Timestamp'].dt.month
    df['Month'] = df['Timestamp'].dt.strftime('%B') # Full month name
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour # Using derived hour for consistency
    df['Day_Name'] = df['Timestamp'].dt.strftime('%A') # Full day name
    df['Message_Length'] = df['Message'].apply(len)
    df['Word_Count'] = df['Message'].apply(lambda s: len(s.split()))
    df['Is_Media'] = df['Message'] == MEDIA_MESSAGE
    df['Is_Link'] = df['Message'].apply(lambda x: bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x))))

    # Emoji count
    def count_emojis(text):
        return emoji.emoji_count(str(text))
    df['Emoji_Count'] = df['Message'].apply(count_emojis)

    return df

def get_cleaned_text(text):
    if pd.isna(text) or text == MEDIA_MESSAGE:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    tokens = nltk.word_tokenize(text)
    # Ensure stopwords are downloaded
    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        st.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        stop_words_list = stopwords.words('english')
    
    # Add custom stopwords if any (e.g., common chat filler words you both use)
    # custom_stopwords = ['hmm', 'ok', 'okay', 'yeah', 'lol']
    # stop_words_list.extend(custom_stopwords)

    tokens = [word for word in tokens if word not in stop_words_list and len(word) > 1] # Min word length
    return " ".join(tokens)

# --- Streamlit App UI ---
st.title(f"💬 Our Chat Story: {AUTHOR_1} & {AUTHOR_2}")
st.markdown("A data-driven journey through our WhatsApp conversations! ❤️")

uploaded_file = st.sidebar.file_uploader("Upload your 'parsed_whatsapp_chat.csv'", type=["csv"])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)

    if df is not None and not df.empty:
        st.sidebar.success(f"Chat data loaded successfully! Analyzing {len(df)} messages.")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Authors:")
        st.sidebar.markdown(f"**You:** {AUTHOR_1}")
        st.sidebar.markdown(f"**Sukyoung:** {AUTHOR_2}")
        st.sidebar.markdown("---")
        # Add date range selector (optional, for very long chats)
        # min_date = df['Date_Only'].min()
        # max_date = df['Date_Only'].max()
        # start_date, end_date = st.sidebar.date_input(
        #     "Select Date Range:",
        #     value=(min_date, max_date),
        #     min_value=min_date,
        #     max_value=max_date
        # )
        # df_filtered = df[(df['Date_Only'] >= start_date) & (df['Date_Only'] <= end_date)]
        df_filtered = df # For now, use all data

        # Create dataframes for each author
        df_author1 = df_filtered[df_filtered['Author'] == AUTHOR_1]
        df_author2 = df_filtered[df_filtered['Author'] == AUTHOR_2]

        # --- Tabs for Analysis ---
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Overall Stats",
            "🕒 Temporal Analysis",
            "📜 Content Deep Dive",
            "💖 Sentiment & Love",
            "✍️ Typing Styles",
            "🖼️ Media & Links",
            "🧠 Vocabulary Richness"
        ])

        # --- Tab 1: Overall Stats ---
        with tab1:
            st.header("Overall Chat Statistics")

            total_messages = len(df_filtered)
            total_media = df_filtered['Is_Media'].sum()
            total_links = df_filtered['Is_Link'].sum()
            total_words = df_filtered['Word_Count'].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", f"{total_messages:,}")
            col2.metric("Total Media Shared", f"{total_media:,}")
            col3.metric("Total Links Shared", f"{total_links:,}")
            col4.metric("Total Words Sent", f"{total_words:,}")

            st.subheader("Message Distribution")
            author_counts = df_filtered['Author'].value_counts()
            fig_msg_dist = px.pie(author_counts, values=author_counts.values, names=author_counts.index,
                                  title="Who Sent More Messages?", hole=0.3)
            fig_msg_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_msg_dist, use_container_width=True)

            st.subheader("Word Contribution")
            word_counts_author = df_filtered.groupby('Author')['Word_Count'].sum()
            fig_word_dist = px.pie(word_counts_author, values=word_counts_author.values, names=word_counts_author.index,
                                   title="Who Wrote More Words?", hole=0.3)
            fig_word_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_word_dist, use_container_width=True)

            st.subheader("Average Message Length (Words)")
            avg_word_count = df_filtered.groupby('Author')['Word_Count'].mean().round(2)
            st.bar_chart(avg_word_count)

        # --- Tab 2: Temporal Analysis ---
        with tab2:
            st.header("When Do We Talk The Most?")

            st.subheader("Messages Over Time")
            messages_over_time = df_filtered.groupby('Date_Only')['Message'].count().reset_index()
            fig_timeline = px.line(messages_over_time, x='Date_Only', y='Message', title="Daily Message Volume")
            fig_timeline.update_xaxes(title_text='Date')
            fig_timeline.update_yaxes(title_text='Number of Messages')
            st.plotly_chart(fig_timeline, use_container_width=True)

            st.subheader("Activity by Hour of Day")
            hourly_activity = df_filtered.groupby(['Hour', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
            # Ensure all hours 0-23 are present for a complete heatmap or bar chart
            all_hours_df = pd.DataFrame({'Hour': range(24)})
            hourly_activity = pd.merge(all_hours_df, hourly_activity, on='Hour', how='left').fillna(0)

            fig_hourly = px.bar(hourly_activity, x='Hour', y=[AUTHOR_1, AUTHOR_2],
                                title="Hourly Chat Activity", barmode='group',
                                labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            st.subheader("Activity by Day of Week")
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_activity = df_filtered.groupby(['Day_Name', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
            daily_activity['Day_Name'] = pd.Categorical(daily_activity['Day_Name'], categories=days_order, ordered=True)
            daily_activity = daily_activity.sort_values('Day_Name')

            fig_daily = px.bar(daily_activity, x='Day_Name', y=[AUTHOR_1, AUTHOR_2],
                               title="Chat Activity by Day of the Week", barmode='group',
                               labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_daily, use_container_width=True)

            st.subheader("Monthly Message Contribution")
            monthly_activity = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
            monthly_activity.sort_values(['Year', 'Month_Num'], inplace=True)
            monthly_activity['Year-Month'] = monthly_activity['Year'].astype(str) + "-" + monthly_activity['Month']
            
            fig_monthly = px.bar(monthly_activity, x='Year-Month', y=[AUTHOR_1, AUTHOR_2],
                                 title="Who Texts More Each Month?", barmode='group',
                                 labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_monthly, use_container_width=True)

        # --- Tab 3: Content Deep Dive ---
        with tab3:
            st.header("What Do We Talk About?")

            # Ensure NLTK resources are available
            try:
                stopwords.words('english')
                nltk.word_tokenize("test")
            except LookupError as e:
                st.error(f"NLTK resource missing: {e}. Please run `nltk.download('stopwords')` and `nltk.download('punkt')` in your Python environment if you haven't.")
                st.stop()

            # Clean text for word clouds and frequency analysis (excluding media)
            df_text_only = df_filtered[~df_filtered['Is_Media']].copy()
            df_text_only['Cleaned_Message'] = df_text_only['Message'].apply(get_cleaned_text)

            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.subheader(f"Word Cloud: {AUTHOR_1}")
                text_author1 = " ".join(df_text_only[df_text_only['Author'] == AUTHOR_1]['Cleaned_Message'])
                if text_author1.strip():
                    wordcloud1 = WordCloud(width=600, height=400, background_color='white', collocations=False).generate(text_author1)
                    fig_wc1, ax_wc1 = plt.subplots()
                    ax_wc1.imshow(wordcloud1, interpolation='bilinear')
                    ax_wc1.axis('off')
                    st.pyplot(fig_wc1)
                else:
                    st.write(f"Not enough text data for {AUTHOR_1} to generate a word cloud.")

            with col_wc2:
                st.subheader(f"Word Cloud: {AUTHOR_2}")
                text_author2 = " ".join(df_text_only[df_text_only['Author'] == AUTHOR_2]['Cleaned_Message'])
                if text_author2.strip():
                    wordcloud2 = WordCloud(width=600, height=400, background_color='white', collocations=False).generate(text_author2)
                    fig_wc2, ax_wc2 = plt.subplots()
                    ax_wc2.imshow(wordcloud2, interpolation='bilinear')
                    ax_wc2.axis('off')
                    st.pyplot(fig_wc2)
                else:
                    st.write(f"Not enough text data for {AUTHOR_2} to generate a word cloud.")

            st.subheader("Most Common Words (After Cleaning)")
            num_common_words = st.slider("Select number of common words to display:", 5, 30, 10)

            all_cleaned_words = " ".join(df_text_only['Cleaned_Message']).split()
            common_words_overall = Counter(all_cleaned_words).most_common(num_common_words)
            if common_words_overall:
                df_common_overall = pd.DataFrame(common_words_overall, columns=['Word', 'Frequency'])
                fig_common_overall = px.bar(df_common_overall, x='Frequency', y='Word', orientation='h', title="Overall Most Common Words")
                st.plotly_chart(fig_common_overall, use_container_width=True)
            else:
                st.write("No common words found (perhaps all messages were media or very short).")


            col_cw1, col_cw2 = st.columns(2)
            with col_cw1:
                st.markdown(f"**Most Common Words: {AUTHOR_1}**")
                words_author1 = " ".join(df_text_only[df_text_only['Author'] == AUTHOR_1]['Cleaned_Message']).split()
                common_words_auth1 = Counter(words_author1).most_common(num_common_words)
                if common_words_auth1:
                    df_common_auth1 = pd.DataFrame(common_words_auth1, columns=['Word', 'Frequency'])
                    fig_common_auth1 = px.bar(df_common_auth1, x='Frequency', y='Word', orientation='h')
                    st.plotly_chart(fig_common_auth1, use_container_width=True)
                else:
                    st.write(f"No common words found for {AUTHOR_1}.")
            with col_cw2:
                st.markdown(f"**Most Common Words: {AUTHOR_2}**")
                words_author2 = " ".join(df_text_only[df_text_only['Author'] == AUTHOR_2]['Cleaned_Message']).split()
                common_words_auth2 = Counter(words_author2).most_common(num_common_words)
                if common_words_auth2:
                    df_common_auth2 = pd.DataFrame(common_words_auth2, columns=['Word', 'Frequency'])
                    fig_common_auth2 = px.bar(df_common_auth2, x='Frequency', y='Word', orientation='h')
                    st.plotly_chart(fig_common_auth2, use_container_width=True)
                else:
                    st.write(f"No common words found for {AUTHOR_2}.")


            st.subheader("Emoji Analysis")
            emoji_counts = {}
            for author in [AUTHOR_1, AUTHOR_2]:
                author_messages = df_filtered[(df_filtered['Author'] == author) & (df_filtered['Emoji_Count'] > 0)]['Message']
                all_emojis_author = []
                for msg in author_messages:
                    all_emojis_author.extend([e['emoji'] for e in emoji.emoji_list(str(msg))])
                emoji_counts[author] = Counter(all_emojis_author)

            col_em1, col_em2 = st.columns(2)
            with col_em1:
                st.markdown(f"**Top Emojis: {AUTHOR_1}**")
                if emoji_counts[AUTHOR_1]:
                    df_emoji1 = pd.DataFrame(emoji_counts[AUTHOR_1].most_common(10), columns=['Emoji', 'Count'])
                    fig_emoji1 = px.bar(df_emoji1, x='Count', y='Emoji', orientation='h')
                    st.plotly_chart(fig_emoji1, use_container_width=True)
                else:
                    st.write(f"{AUTHOR_1} hasn't used many emojis or none were found.")
            with col_em2:
                st.markdown(f"**Top Emojis: {AUTHOR_2}**")
                if emoji_counts[AUTHOR_2]:
                    df_emoji2 = pd.DataFrame(emoji_counts[AUTHOR_2].most_common(10), columns=['Emoji', 'Count'])
                    fig_emoji2 = px.bar(df_emoji2, x='Count', y='Emoji', orientation='h')
                    st.plotly_chart(fig_emoji2, use_container_width=True)
                else:
                    st.write(f"{AUTHOR_2} hasn't used many emojis or none were found.")


        # --- Tab 4: Sentiment & Love ---
        with tab4:
            st.header("Sentiment & Expressions of Love")

            # Ensure NLTK VADER is available
            try:
                analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                st.info("Downloading NLTK VADER lexicon...")
                nltk.download('vader_lexicon')
                analyzer = SentimentIntensityAnalyzer()

            df_text_only_sentiment = df_filtered[~df_filtered['Is_Media']].copy() # Use original text for VADER
            
            if not df_text_only_sentiment.empty:
                df_text_only_sentiment['Sentiment_Score'] = df_text_only_sentiment['Message'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
                
                def sentiment_category(score):
                    if score > 0.05: return 'Positive'
                    elif score < -0.05: return 'Negative'
                    else: return 'Neutral'
                df_text_only_sentiment['Sentiment_Type'] = df_text_only_sentiment['Sentiment_Score'].apply(sentiment_category)

                st.subheader("Overall Sentiment Distribution")
                sentiment_counts = df_text_only_sentiment['Sentiment_Type'].value_counts()
                fig_sentiment_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                                        title="Overall Sentiment of Messages", hole=0.3)
                st.plotly_chart(fig_sentiment_pie, use_container_width=True)

                st.subheader("Sentiment by Author")
                sentiment_by_author = df_text_only_sentiment.groupby('Author')['Sentiment_Score'].mean().reset_index()
                fig_sentiment_author = px.bar(sentiment_by_author, x='Author', y='Sentiment_Score', color='Author',
                                            title="Average Sentiment Score per Author (Higher is more positive)")
                st.plotly_chart(fig_sentiment_author, use_container_width=True)

                st.subheader("Sentiment Over Time")
                sentiment_over_time = df_text_only_sentiment.groupby('Date_Only')['Sentiment_Score'].mean().reset_index()
                fig_sentiment_timeline = px.line(sentiment_over_time, x='Date_Only', y='Sentiment_Score', title="Average Daily Sentiment")
                fig_sentiment_timeline.add_hline(y=0, line_dash="dot", annotation_text="Neutral")
                st.plotly_chart(fig_sentiment_timeline, use_container_width=True)
            else:
                st.write("Not enough text messages to perform sentiment analysis.")


            st.subheader("Expressions of Love ❤️")
            def count_love_keywords(message):
                message_lower = str(message).lower()
                count = 0
                for keyword in LOVE_KEYWORDS:
                    if keyword.lower() in message_lower:
                        count += 1
                return count > 0 # True if any love keyword is found

            df_filtered['Has_Love_Keyword'] = df_filtered['Message'].apply(count_love_keywords)
            love_expressions_count = df_filtered.groupby('Author')['Has_Love_Keyword'].sum()

            if not love_expressions_count.empty:
                fig_love = px.bar(love_expressions_count, x=love_expressions_count.index, y=love_expressions_count.values,
                                color=love_expressions_count.index,
                                title="Who Expresses 'Love' More (Keyword Based)?",
                                labels={'y': 'Number of Messages with Love Keywords', 'index': 'Author'})
                st.plotly_chart(fig_love, use_container_width=True)

                st.markdown(f"**Keywords used:** `{', '.join(LOVE_KEYWORDS[:10])}...` (and more)")
                st.caption("Note: This is a simple keyword count and might not capture all nuances.")
            else:
                st.write("No 'love' keywords found based on the defined list.")

        # --- Tab 5: Typing Styles ---
        with tab5:
            st.header("Evolution of Typing Styles")

            st.subheader("Average Message Length (Words) Over Time")
            avg_len_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Word_Count'].mean().unstack(fill_value=0).reset_index()
            avg_len_time.sort_values(['Year', 'Month_Num'], inplace=True)
            avg_len_time['Year-Month'] = avg_len_time['Year'].astype(str) + "-" + avg_len_time['Month']

            fig_avg_len_time = px.line(avg_len_time, x='Year-Month', y=[AUTHOR_1, AUTHOR_2],
                                       title="Average Message Length (Words) per Month",
                                       labels={'value': 'Average Words per Message', 'variable': 'Author'})
            st.plotly_chart(fig_avg_len_time, use_container_width=True)

            st.subheader("Emoji Usage Over Time (Total Emojis Sent)")
            emoji_usage_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Emoji_Count'].sum().unstack(fill_value=0).reset_index()
            emoji_usage_time.sort_values(['Year', 'Month_Num'], inplace=True)
            emoji_usage_time['Year-Month'] = emoji_usage_time['Year'].astype(str) + "-" + emoji_usage_time['Month']
            
            fig_emoji_time = px.line(emoji_usage_time, x='Year-Month', y=[AUTHOR_1, AUTHOR_2],
                                    title="Total Emojis Used per Month",
                                    labels={'value': 'Total Emojis', 'variable': 'Author'})
            st.plotly_chart(fig_emoji_time, use_container_width=True)

        # --- Tab 6: Media & Links ---
        with tab6:
            st.header("Media and Link Sharing")

            st.subheader("Media Shared")
            media_counts_author = df_filtered.groupby('Author')['Is_Media'].sum()
            fig_media_dist = px.pie(media_counts_author, values=media_counts_author.values, names=media_counts_author.index,
                                   title="Who Shared More Media?", hole=0.3)
            fig_media_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_media_dist, use_container_width=True)

            st.subheader("Links Shared")
            link_counts_author = df_filtered.groupby('Author')['Is_Link'].sum()
            fig_link_dist = px.pie(link_counts_author, values=link_counts_author.values, names=link_counts_author.index,
                                   title="Who Shared More Links?", hole=0.3)
            fig_link_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_link_dist, use_container_width=True)

            st.subheader("When is Media Shared Most?")
            media_by_hour = df_filtered[df_filtered['Is_Media']].groupby(['Hour', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
            all_hours_df_media = pd.DataFrame({'Hour': range(24)})
            media_by_hour = pd.merge(all_hours_df_media, media_by_hour, on='Hour', how='left').fillna(0)
            fig_media_hourly = px.bar(media_by_hour, x='Hour', y=[AUTHOR_1, AUTHOR_2],
                                      title="Hourly Media Sharing Activity", barmode='group',
                                      labels={'value': 'Number of Media Messages', 'variable': 'Author'})
            st.plotly_chart(fig_media_hourly, use_container_width=True)

        with tab7:
            st.header("🧠 Vocabulary Richness")

            df_text = df_filtered[~df_filtered['Is_Media']].copy()
            df_text['Cleaned'] = df_text['Message'].apply(get_cleaned_text)

            vocab_stats = []
            for author in [AUTHOR_1, AUTHOR_2]:
                messages = " ".join(df_text[df_text['Author'] == author]['Cleaned'])
                tokens = word_tokenize(messages)
                unique_tokens = set(tokens)
                vocab_richness = len(unique_tokens) / len(tokens) if tokens else 0
                vocab_stats.append({
                    'Author': author,
                    'Total Words': len(tokens),
                    'Unique Words': len(unique_tokens),
                    'Vocabulary Richness (TTR)': round(vocab_richness, 3)
                })

            st.dataframe(pd.DataFrame(vocab_stats))

            st.subheader("Top Unique Words")
            top_n = st.slider("Select number of top unique words to display", 5, 50, 20)
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.markdown(f"**{AUTHOR_1}**")
                tokens1 = word_tokenize(" ".join(df_text[df_text['Author'] == AUTHOR_1]['Cleaned']))
                freq1 = FreqDist(tokens1)
                common1 = pd.DataFrame(freq1.most_common(top_n), columns=['Word', 'Frequency'])
                st.dataframe(common1)

            with col_v2:
                st.markdown(f"**{AUTHOR_2}**")
                tokens2 = word_tokenize(" ".join(df_text[df_text['Author'] == AUTHOR_2]['Cleaned']))
                freq2 = FreqDist(tokens2)
                common2 = pd.DataFrame(freq2.most_common(top_n), columns=['Word', 'Frequency'])
                st.dataframe(common2)
        # --- End of Tabs ---    
        

    elif df is not None and df.empty and uploaded_file:
        st.warning("The uploaded CSV was processed but resulted in an empty dataset. This might be due to author name mismatches or no relevant messages.")
    # else: # df is None (handled by load_and_preprocess_data error messages)
    #    pass

else:
    st.info("Awaiting your WhatsApp chat CSV file... 😊")

st.sidebar.markdown("---")
