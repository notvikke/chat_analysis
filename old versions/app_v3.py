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
from langdetect import detect, DetectorFactory, LangDetectException # Add LangDetectException

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
    'heart', 'hearts', ' babe', 'baby', 'darling', 'sweetheart', 'honey','cutie','cute',
    '❤️', '😘', '😍', '🥰', '💕', '💖', '💗', '💓', '💞', '💘', '💌'
] # Add your own specific terms!
SLANG_NICKNAMES = {
    AUTHOR_1: ['bro', 'dude', 'my dude', 'man'], # Your slang/nicknames for Sukyoung or general
    AUTHOR_2: ['jagiya', 'oppa', 'sweetie pie', 'dear'], # Sukyoung's slang/nicknames for you or general
    'common': ['lol', 'lmao', 'omg', 'brb', 'ttyl', 'wyd', 'wbu', 'ikr', 'gg', 'afaik', 'btw'] # Common slang
}
GREETING_CELEBRATION_TERMS = [
    'happy birthday', 'hbd', 'happy anniversary', 'merry christmas', 'happy new year',
    'congrats', 'congratulations', 'good morning', 'good night', 'good evening', 'congratzzz',
    'yay', 'hooray', 'celebrate', 'happy bday', 'happy new yr'
]

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

    try:
        df['Timestamp'] = pd.to_datetime(df['DateTime_Str'])
    except Exception as e:
        st.error(f"Error parsing DateTime_Str column: {e}")
        st.info("Expected format: YYYY-MM-DD HH:MM:SS")
        return None

    df.dropna(subset=['Message', 'Author'], inplace=True)
    df['Author'] = df['Author'].str.strip()
    df['Message'] = df['Message'].astype(str).str.strip() # Ensure Message is string

    df = df[df['Author'].isin([AUTHOR_1, AUTHOR_2])]
    if df.empty:
        st.warning(f"No messages found for authors '{AUTHOR_1}' or '{AUTHOR_2}'. Please check author names in your CSV.")
        return None

    # Ensure DataFrame is sorted by Timestamp early for correct subsequent calculations
    df.sort_values('Timestamp', inplace=True, ignore_index=True)

    df['Date_Only'] = df['Timestamp'].dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Num'] = df['Timestamp'].dt.month
    df['Month'] = df['Timestamp'].dt.strftime('%B')
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_Name'] = df['Timestamp'].dt.strftime('%A')
    # Renaming 'Message_Length' to 'Message_Length_Chars' for consistency with later tabs
    df['Message_Length_Chars'] = df['Message'].apply(len)
    df['Word_Count'] = df['Message'].apply(lambda s: len(s.split()))
    df['Is_Media'] = df['Message'] == MEDIA_MESSAGE
    df['Is_Link'] = df['Message'].apply(lambda x: bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x))))

    def count_emojis_func(text): # Renamed to avoid conflict if 'count_emojis' is a global var
        return emoji.emoji_count(str(text))
    df['Emoji_Count'] = df['Message'].apply(count_emojis_func)

    # --- INSERTED MISSING LINGUISTIC AND CONVERSATION DYNAMICS FEATURES ---
    df['Punctuation_Count'] = df['Message'].apply(lambda s: len(re.findall(r'[!?\.]', str(s))))
    df['Exclamation_Count'] = df['Message'].apply(lambda s: str(s).count('!'))
    df['Question_Count'] = df['Message'].apply(lambda s: str(s).count('?'))
    df['Is_Caps'] = df['Message'].apply(lambda s: str(s).isupper() and len(str(s)) > 3)
    df['Elongations'] = df['Message'].apply(lambda s: len(re.findall(r'(\w)\1{2,}', str(s))))

    def detect_lang_safe(text):
        if pd.isna(text) or text == MEDIA_MESSAGE or len(str(text).strip()) < 10:
            return 'unknown'
        try:
            return detect(str(text))
        except LangDetectException: # Catch specific langdetect error
            return 'error_langdetect'
        except Exception: # Catch any other potential errors
            return 'error_other_lang'
    df['Language'] = df['Message'].apply(detect_lang_safe)

    df['Response_Time_Seconds'] = pd.NA
    df['Is_Initiator'] = False
    df['Consecutive_Count'] = 0

    conversation_break_threshold = pd.Timedelta(minutes=30)
    last_author_for_consecutive = None

    for i, row in df.iterrows():
        if i > 0:
            prev_row = df.loc[i-1]
            time_diff = row['Timestamp'] - prev_row['Timestamp']
            if row['Author'] != prev_row['Author']:
                df.loc[i, 'Response_Time_Seconds'] = time_diff.total_seconds()
            if time_diff > conversation_break_threshold:
                df.loc[i, 'Is_Initiator'] = True
        else:
            df.loc[i, 'Is_Initiator'] = True

        if row['Author'] == last_author_for_consecutive:
            df.loc[i, 'Consecutive_Count'] = (df.loc[i-1, 'Consecutive_Count'] + 1) if i > 0 else 1
        else:
            df.loc[i, 'Consecutive_Count'] = 1
        last_author_for_consecutive = row['Author']
    # --- END OF INSERTED BLOCK ---

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

# --- Helper Functions ---
# ... (other helper functions like load_and_preprocess_data, get_cleaned_text) ...

def calculate_ttr(texts_list):
    """
    Calculates the Type-Token Ratio for a list of cleaned text strings.
    TTR = (Number of Unique Words / Total Number of Words) * 100
    """
    if not texts_list:  # Handle empty list
        return 0.0
    
    all_words = []
    for text_item in texts_list:
        # Ensure text_item is a string and split into words
        # This handles cases where a list might contain non-string items or empty strings
        if isinstance(text_item, str) and text_item.strip():
            all_words.extend(text_item.split())
            
    if not all_words:  # Handle case where all texts were empty or non-strings
        return 0.0
    
    unique_words_count = len(set(all_words))
    total_words_count = len(all_words)
    
    return (unique_words_count / total_words_count) * 100 if total_words_count > 0 else 0.0

# ... (rest of your helper functions and then the Streamlit UI code) ...

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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
            "📊 Overall Stats",
            "🕒 Temporal Analysis",
            "📜 Content Deep Dive",
            "💖 Sentiment & Love",
            "✍️ Typing Styles",
            "🖼️ Media & Links",
            "🧠 Vocabulary Richness",
            "🔤 Linguistic Style",
            "💬 Conversation Dynamics",
            "📈 Trends & Patterns"
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
        
        
        with tab8: # Corresponds to "✍️ Linguistic Styles"
            st.header("✍️ Our Linguistic Styles")

            st.subheader("Typing Style Indicators")
            col_ls1, col_ls2 = st.columns(2)
            with col_ls1:
                st.metric(f"Total Exclamations by {AUTHOR_1}", df_filtered[df_filtered['Author'] == AUTHOR_1]['Exclamation_Count'].sum())
                st.metric(f"Total Questions by {AUTHOR_1}", df_filtered[df_filtered['Author'] == AUTHOR_1]['Question_Count'].sum())
                st.metric(f"Total Elongations by {AUTHOR_1}", df_filtered[df_filtered['Author'] == AUTHOR_1]['Elongations'].sum())
                st.metric(f"Total CAPS Msgs by {AUTHOR_1}", df_filtered[df_filtered['Author'] == AUTHOR_1]['Is_Caps'].sum())
            with col_ls2:
                st.metric(f"Total Exclamations by {AUTHOR_2}", df_filtered[df_filtered['Author'] == AUTHOR_2]['Exclamation_Count'].sum())
                st.metric(f"Total Questions by {AUTHOR_2}", df_filtered[df_filtered['Author'] == AUTHOR_2]['Question_Count'].sum())
                st.metric(f"Total Elongations by {AUTHOR_2}", df_filtered[df_filtered['Author'] == AUTHOR_2]['Elongations'].sum())
                st.metric(f"Total CAPS Msgs by {AUTHOR_2}", df_filtered[df_filtered['Author'] == AUTHOR_2]['Is_Caps'].sum())


            st.subheader("Message Length Evolution (Characters)")
            avg_len_chars_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Message_Length_Chars'].mean().unstack().reset_index()
            avg_len_chars_time.sort_values(['Year', 'Month_Num'], inplace=True)
            avg_len_chars_time['Year-Month'] = avg_len_chars_time['Year'].astype(str) + "-" + avg_len_chars_time['Month']
            # Ensure authors exist in columns before plotting to avoid KeyError
            authors_present_ml = [col for col in [AUTHOR_1, AUTHOR_2] if col in avg_len_chars_time.columns]
            if authors_present_ml:
                fig_avg_len_chars_time = px.line(avg_len_chars_time, x='Year-Month', y=authors_present_ml,
                                        title="Avg Message Length (Characters) per Month",
                                        labels={'value': 'Avg Chars per Message', 'variable': 'Author'})
                st.plotly_chart(fig_avg_len_chars_time, use_container_width=True)
            else:
                st.write("Not enough data to plot message length evolution for one or both authors.")


            st.subheader("Vocabulary Richness (Type-Token Ratio - TTR) per Month")
            df_text_only_ttr = df_filtered[~df_filtered['Is_Media']].copy()
            # Assuming get_cleaned_text is defined
            df_text_only_ttr['Cleaned_Message_TTR'] = df_text_only_ttr['Message'].apply(get_cleaned_text)

            ttr_data = []
            for author_name_ttr in [AUTHOR_1, AUTHOR_2]:
                author_df_ttr = df_text_only_ttr[df_text_only_ttr['Author'] == author_name_ttr]
                for (year, month_name, month_num), group in author_df_ttr.groupby(['Year', 'Month', 'Month_Num']):
                    # Assuming calculate_ttr is defined
                    ttr = calculate_ttr(group['Cleaned_Message_TTR'].tolist())
                    ttr_data.append({'Year-Month': f"{year}-{month_name}",
                                    'Year': year, 'Month_Num': month_num,
                                    'Author': author_name_ttr, 'TTR': ttr})

            if ttr_data:
                df_ttr = pd.DataFrame(ttr_data).sort_values(['Year', 'Month_Num'])
                fig_ttr = px.line(df_ttr, x='Year-Month', y='TTR', color='Author',
                                title="Vocabulary Richness (TTR %) Over Time",
                                labels={'TTR': 'TTR (%)', 'variable': 'Author'})
                st.plotly_chart(fig_ttr, use_container_width=True)
            else:
                st.write("Not enough data to calculate TTR over time.")


            st.subheader("Language Use (Detected)")
            lang_counts_author = df_filtered.groupby(['Author', 'Language'])['Message'].count().unstack(fill_value=0)
            if not lang_counts_author.empty:
                st.write("Detected languages per author (for messages > 10 chars):")
                st.dataframe(lang_counts_author) # Show raw counts

                # Get top N languages overall, excluding 'unknown' and 'error' types for plotting
                all_langs_sum = lang_counts_author.sum().sort_values(ascending=False)
                common_langs_to_plot = [lang for lang in all_langs_sum.index
                                        if lang not in ['unknown', 'error_langdetect', 'error_unknown']][:5] # Top 5 actual languages

                if common_langs_to_plot:
                    lang_counts_filtered_plot = lang_counts_author[common_langs_to_plot].reset_index()
                    lang_melted_plot = lang_counts_filtered_plot.melt(id_vars='Author', value_vars=common_langs_to_plot,
                                                                var_name='Language', value_name='Count')
                    if not lang_melted_plot.empty and lang_melted_plot['Count'].sum() > 0:
                        fig_lang = px.bar(lang_melted_plot, x='Author', y='Count', color='Language', barmode='group',
                                        title=f"Top Detected Languages Usage (Max 5)")
                        st.plotly_chart(fig_lang, use_container_width=True)
                    else:
                        st.write("No significant usage of common detected languages to plot.")
                else:
                    st.write("No common languages (other than 'unknown' or errors) detected for plotting.")
            else:
                st.write("Language detection data not available.")


            st.subheader("Slang / Nickname Usage")
            # Assuming SLANG_NICKNAMES constant is defined
            slang_data = []
            for author_name_slang, slang_list_user in SLANG_NICKNAMES.items():
                if author_name_slang == 'common': continue
                for slang_term in slang_list_user:
                    count = df_filtered[(df_filtered['Author'] == author_name_slang) & (df_filtered['Message'].str.contains(slang_term, case=False, na=False))].shape[0]
                    if count > 0: # Only add if used
                        slang_data.append({'Author': author_name_slang, 'Term': slang_term, 'Count': count})

            for slang_term_common in SLANG_NICKNAMES.get('common', []):
                count_auth1 = df_filtered[(df_filtered['Author'] == AUTHOR_1) & (df_filtered['Message'].str.contains(slang_term_common, case=False, na=False))].shape[0]
                if count_auth1 > 0:
                    slang_data.append({'Author': AUTHOR_1, 'Term': f"{slang_term_common} (common)", 'Count': count_auth1})
                count_auth2 = df_filtered[(df_filtered['Author'] == AUTHOR_2) & (df_filtered['Message'].str.contains(slang_term_common, case=False, na=False))].shape[0]
                if count_auth2 > 0:
                    slang_data.append({'Author': AUTHOR_2, 'Term': f"{slang_term_common} (common)", 'Count': count_auth2})

            if slang_data:
                df_slang = pd.DataFrame(slang_data)
                df_slang = df_slang[df_slang['Count'] > 0].sort_values('Count', ascending=False) # Ensure count > 0
                if not df_slang.empty:
                    fig_slang = px.bar(df_slang, x='Term', y='Count', color='Author', barmode='group',
                                    title="Use of Defined Slang/Nicknames")
                    st.plotly_chart(fig_slang, use_container_width=True)
                else:
                    st.write("None of the defined slang/nicknames found.")
            else:
                st.write("No slang/nicknames data to display. Define them in `SLANG_NICKNAMES`.")
            st.caption("Define your slang/nicknames in the `SLANG_NICKNAMES` list in the script for accurate results.")

        # --- Tab 6: Conversation Dynamics ---
        with tab9: # Corresponds to "🔄 Conversation Dynamics"
            st.header("🔄 Our Conversation Dynamics")

            st.subheader("Average Response Time (Seconds)")
            valid_response_times = df_filtered[df_filtered['Response_Time_Seconds'].notna() & (df_filtered['Response_Time_Seconds'] > 0)]
            if not valid_response_times.empty:
                avg_response_time_author = valid_response_times.groupby('Author')['Response_Time_Seconds'].mean().round(2)
                if not avg_response_time_author.empty:
                    st.bar_chart(avg_response_time_author)
                    st.caption("Response time is calculated when a user sends a message after the other user's last message (responses > 0s).")
                else:
                    st.write("Not enough data to calculate average response times for authors.")
            else:
                st.write("No valid response time data available.")


            st.subheader("Who Initiates Conversations More?")
            initiator_counts = df_filtered[df_filtered['Is_Initiator']].groupby('Author')['Message'].count()
            if not initiator_counts.empty:
                fig_initiator = px.pie(initiator_counts, values=initiator_counts.values, names=initiator_counts.index,
                                    title="Conversation Initiators (Message after >30min silence)", hole=0.3)
                fig_initiator.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_initiator, use_container_width=True)
            else:
                st.write("Could not determine conversation initiators (e.g., all messages within 30 mins of each other).")


            st.subheader("Consecutive Messages (Message Streaks)")
            max_consecutive_author = df_filtered.groupby('Author')['Consecutive_Count'].max()
            avg_consecutive_author = df_filtered[df_filtered['Consecutive_Count'] > 1].groupby('Author')['Consecutive_Count'].mean().round(2)

            col_cons1, col_cons2 = st.columns(2)
            with col_cons1:
                st.metric(f"Longest Streak by {AUTHOR_1}", max_consecutive_author.get(AUTHOR_1, 0))
                st.metric(f"Avg. Streak Length by {AUTHOR_1}", avg_consecutive_author.get(AUTHOR_1, 0.0)) # Default to 0.0 if no streaks > 1
            with col_cons2:
                st.metric(f"Longest Streak by {AUTHOR_2}", max_consecutive_author.get(AUTHOR_2, 0))
                st.metric(f"Avg. Streak Length by {AUTHOR_2}", avg_consecutive_author.get(AUTHOR_2, 0.0)) # Default to 0.0
            st.caption("A streak is a series of messages sent by one person before the other replies. Avg. streak length considers streaks > 1 message.")

        # --- Tab 7: Special Content Detection ---
        with tab10: # Corresponds to "🎉 Special Content"
            st.header("🎉 Special Content & Moments")

            st.subheader("Expressions of Love ❤️ (Keyword Based)")
            # Assuming LOVE_KEYWORDS is defined
            df_filtered['Has_Love_Keyword'] = df_filtered['Message'].apply(
                lambda x: any(keyword.lower() in str(x).lower() for keyword in LOVE_KEYWORDS)
            )
            love_expressions_count = df_filtered.groupby('Author')['Has_Love_Keyword'].sum()

            if not love_expressions_count.empty and love_expressions_count.sum() > 0 :
                fig_love = px.bar(love_expressions_count, x=love_expressions_count.index, y=love_expressions_count.values,
                                color=love_expressions_count.index,
                                title="Who Expresses 'Love' More (Keyword Based)?",
                                labels={'y': 'Number of Messages with Love Keywords', 'index': 'Author'})
                st.plotly_chart(fig_love, use_container_width=True)
            else:
                st.write("No 'love' keywords found based on the defined list, or no messages matching.")
            st.markdown(f"**Keywords tracked (first 10):** `{', '.join(LOVE_KEYWORDS[:10])}...` (Customize this list!)")


            st.subheader("Peak Messaging Days (Potential Important Dates)")
            daily_msg_counts = df_filtered.groupby('Date_Only')['Message'].count().reset_index()
            daily_msg_counts.rename(columns={'Message': 'Total_Messages'}, inplace=True)
            daily_msg_counts.sort_values('Total_Messages', ascending=False, inplace=True)

            num_peak_days = st.slider("Number of Peak Days to Show:", 1, min(20, len(daily_msg_counts)), 5, key="peak_days_slider")
            if not daily_msg_counts.empty:
                st.dataframe(daily_msg_counts.head(num_peak_days))
                st.caption("These are days with the highest message volume. Check them for anniversaries, birthdays, etc.")

                st.subheader("Greeting/Celebration Term Detection on Peak Days")
                # Assuming GREETING_CELEBRATION_TERMS is defined
                peak_dates_list = daily_msg_counts.head(num_peak_days)['Date_Only'].tolist()
                df_peak_days_msgs = df_filtered[df_filtered['Date_Only'].isin(peak_dates_list)].copy()

                def find_greeting_terms(message):
                    found_terms = [term for term in GREETING_CELEBRATION_TERMS if term.lower() in str(message).lower()]
                    return ", ".join(found_terms) if found_terms else None

                df_peak_days_msgs['Found_Greetings'] = df_peak_days_msgs['Message'].apply(find_greeting_terms)
                greeting_msgs_on_peak = df_peak_days_msgs[df_peak_days_msgs['Found_Greetings'].notna()]

                if not greeting_msgs_on_peak.empty:
                    st.write("Messages with greeting/celebration terms on these peak days:")
                    st.dataframe(greeting_msgs_on_peak[['Date_Only', 'Author', 'Message', 'Found_Greetings']])
                else:
                    st.write("No predefined greeting/celebration terms found on the selected peak days.")
                st.caption(f"**Terms searched (first 5):** `{', '.join(GREETING_CELEBRATION_TERMS[:5])}...` (Customize list)")
            else:
                st.write("Not enough data to determine peak messaging days.")

                        # --- End of Tabs ---    
        

    elif df is not None and df.empty and uploaded_file:
        st.warning("The uploaded CSV was processed but resulted in an empty dataset. This might be due to author name mismatches or no relevant messages.")
    # else: # df is None (handled by load_and_preprocess_data error messages)
    #    pass

else:
    st.info("Awaiting your WhatsApp chat CSV file... 😊")

st.sidebar.markdown("---")
