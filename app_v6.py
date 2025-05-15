import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from nltk import word_tokenize
from langdetect import detect, DetectorFactory, LangDetectException
import random # For Fun Fact button
from io import StringIO # For caching uploaded file content

DetectorFactory.seed = 0
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    st.info("Downloading NLTK 'punkt' resource...") # Optional: inform the user
    nltk.download('punkt', quiet=True) # quiet=True suppresses verbose output
except LookupError: # More general NLTK data lookup error
    st.info("Downloading NLTK 'punkt' resource (LookupError)...")
    nltk.download('punkt', quiet=True)

def download_nltk_resources():
    resources = {"stopwords": "corpora/stopwords", "vader_lexicon": "sentiment/vader_lexicon.zip", "punkt": "tokenizers/punkt"}
    for name, path in resources.items():
        try: nltk.data.find(path)
        except LookupError:
            print(f"NLTK resource '{name}' not found. Downloading...")
            try: nltk.download(name, quiet=True); print(f"NLTK resource '{name}' downloaded.")
            except Exception as e: print(f"ERROR: Failed to download NLTK resource '{name}': {e}")
download_nltk_resources()

st.set_page_config(layout="wide", page_title="Vikke & Sukyoung's Interactive Chat Story")

# --- Global Variables / Constants ---
AUTHOR_1 = "Vikke"
AUTHOR_2 = "Sukyoung"

MEDIA_MESSAGE = "<Media omitted>"
LOVE_KEYWORDS = [
    'love', 'miss u', 'miss you', 'luv u', 'luv you', 'ily', 'i love you', 'my love', 'heart', 'hearts',
    ' babe', 'baby', 'darling', 'sweetheart', 'honey','cutie','cute', '❤️', '😘', '😍', '🥰', '💕', '💖',
    '💗', '💓', '💞', '💘', '💌'
]
SLANG_NICKNAMES = {
    AUTHOR_1: ['bro', 'dude', 'my dude', 'man'],
    AUTHOR_2: ['jagiya', 'oppa', 'sweetie pie', 'dear'],
    'common': ['lol', 'lmao', 'omg', 'brb', 'ttyl', 'wyd', 'wbu', 'ikr', 'gg', 'afaik', 'btw']
}
GREETING_CELEBRATION_TERMS = [
    'happy birthday', 'hbd', 'happy anniversary', 'merry christmas', 'happy new year', 'congrats',
    'congratulations', 'good morning', 'good night', 'good evening', 'congratzzz', 'yay', 'hooray',
    'celebrate', 'happy bday', 'happy new yr'
]

# --- Caching Data Loading and Initial Preprocessing ---
@st.cache_data
def load_initial_data(uploaded_file_content_str):
    try:
        df = pd.read_csv(StringIO(uploaded_file_content_str), header=None, names=[
            'DateTime_Str', 'Author', 'Message', 'Date_Only_Str',
            'Time_Only_Str', 'Hour_Parsed', 'Day_Name_Parsed'
        ])
    except Exception as e: st.error(f"Error loading CSV: {e}"); return None
    try: df['Timestamp'] = pd.to_datetime(df['DateTime_Str'])
    except Exception as e: st.error(f"Error parsing DateTime_Str: {e}"); return None

    df.dropna(subset=['Message', 'Author'], inplace=True)
    df['Author'] = df['Author'].str.strip()
    df['Message'] = df['Message'].astype(str).str.strip()
    
    df = df[df['Author'].isin([AUTHOR_1, AUTHOR_2])]
    if df.empty:
        st.warning(f"No messages found for authors '{AUTHOR_1}' or '{AUTHOR_2}'.")
        return None
    
    df.sort_values('Timestamp', inplace=True, ignore_index=True)
    df['Date_Only'] = df['Timestamp'].dt.date
    df['Year'] = df['Timestamp'].dt.year
    df['Month_Num'] = df['Timestamp'].dt.month
    df['Month'] = df['Timestamp'].dt.strftime('%B')
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_Name'] = df['Timestamp'].dt.strftime('%A')
    df['Message_Length_Chars'] = df['Message'].apply(len)
    df['Word_Count'] = df['Message'].apply(lambda s: len(s.split()))
    df['Is_Media'] = df['Message'] == MEDIA_MESSAGE
    df['Is_Link'] = df['Message'].apply(lambda x: bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(x))))
    def count_emojis_func(text): return emoji.emoji_count(str(text))
    df['Emoji_Count'] = df['Message'].apply(count_emojis_func)
    df['Punctuation_Count'] = df['Message'].apply(lambda s: len(re.findall(r'[!?\.]', str(s))))
    df['Exclamation_Count'] = df['Message'].apply(lambda s: str(s).count('!'))
    df['Question_Count'] = df['Message'].apply(lambda s: str(s).count('?'))
    df['Is_Caps'] = df['Message'].apply(lambda s: str(s).isupper() and len(str(s)) > 3)
    df['Elongations'] = df['Message'].apply(lambda s: len(re.findall(r'(\w)\1{2,}', str(s))))
    def detect_lang_safe(text):
        if pd.isna(text) or text == MEDIA_MESSAGE or len(str(text).strip()) < 10: return 'unknown'
        try: return detect(str(text))
        except LangDetectException: return 'error_langdetect'
        except Exception: return 'error_other_lang'
    df['Language'] = df['Message'].apply(detect_lang_safe)
    return df

# --- Function for dynamic features based on user input ---
def calculate_dynamic_features(df_input, silence_thresh_minutes):
    df = df_input.copy()
    df['Response_Time_Seconds'] = pd.NA
    df['Is_Initiator'] = False
    df['Consecutive_Count'] = 0
    conversation_break_threshold = pd.Timedelta(minutes=silence_thresh_minutes)
    last_author_for_consecutive = None
    df.reset_index(drop=True, inplace=True)
    for i, row in df.iterrows():
        if i > 0:
            prev_row = df.loc[i-1]
            time_diff = row['Timestamp'] - prev_row['Timestamp']
            if row['Author'] != prev_row['Author']: df.loc[i, 'Response_Time_Seconds'] = time_diff.total_seconds()
            if time_diff > conversation_break_threshold: df.loc[i, 'Is_Initiator'] = True
        else: df.loc[i, 'Is_Initiator'] = True
        if row['Author'] == last_author_for_consecutive:
            df.loc[i, 'Consecutive_Count'] = (df.loc[i-1, 'Consecutive_Count'] + 1) if i > 0 else 1
        else: df.loc[i, 'Consecutive_Count'] = 1
        last_author_for_consecutive = row['Author']
    return df

def get_cleaned_text(text, custom_stopwords=None):
    if pd.isna(text) or text == MEDIA_MESSAGE: return ""
    text = str(text).lower(); text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    try: stop_words_list = set(stopwords.words('english'))
    except LookupError: st.error("NLTK stopwords not found."); return " ".join(tokens)
    if custom_stopwords: stop_words_list.update(custom_stopwords)
    tokens = [word for word in tokens if word not in stop_words_list and len(word) > 1]
    return " ".join(tokens)

def calculate_ttr(texts_list):
    if not texts_list: return 0.0
    all_words = [word for text_item in texts_list if isinstance(text_item, str) and text_item.strip() for word in text_item.split()]
    if not all_words: return 0.0
    return (len(set(all_words)) / len(all_words)) * 100 if len(all_words) > 0 else 0.0

# --- Main App ---
st.title(f"💬 Our Chat Story: {AUTHOR_1} & {AUTHOR_2}")
st.markdown("A data-driven journey through our WhatsApp conversations! ❤️")

df_initial = None
uploaded_file = st.sidebar.file_uploader("Upload your 'parsed_whatsapp_chat.csv'", type=["csv"])

if uploaded_file:
    uploaded_file_content_str = uploaded_file.getvalue().decode('utf-8')
    df_initial = load_initial_data(uploaded_file_content_str)

if df_initial is not None and not df_initial.empty:
    st.sidebar.success(f"Chat data loaded! Analyzing {len(df_initial)} messages.")
    st.sidebar.markdown("---"); st.sidebar.subheader("Authors:")
    st.sidebar.markdown(f"**You:** {AUTHOR_1}")
    st.sidebar.markdown(f"**Sukyoung:** {AUTHOR_2}")
    st.sidebar.markdown("---")

    st.sidebar.subheader("⚙️ Analysis Controls")
    min_date_available = df_initial['Date_Only'].min()
    max_date_available = df_initial['Date_Only'].max()
    selected_start_date, selected_end_date = st.sidebar.date_input(
        "Select Date Range for Analysis:", value=(min_date_available, max_date_available),
        min_value=min_date_available, max_value=max_date_available, key="date_range_sidebar"
    )
    silence_threshold_minutes_input = st.sidebar.slider(
        "Silence Threshold for New Conversation (minutes):", min_value=5, max_value=180, value=30, step=5,
        key="silence_slider_sidebar", help="Defines pause indicating a new conversation."
    )
    df_filtered_by_date = df_initial[
        (df_initial['Date_Only'] >= selected_start_date) & (df_initial['Date_Only'] <= selected_end_date)
    ].copy()
    df_filtered = calculate_dynamic_features(df_filtered_by_date, silence_threshold_minutes_input)
    
    df_author1 = df_filtered[df_filtered['Author'] == AUTHOR_1]
    df_author2 = df_filtered[df_filtered['Author'] == AUTHOR_2]

    st.sidebar.markdown("---")
    if st.sidebar.button("🤯 Surprise Me with a Fun Fact!", key="fun_fact_btn_sidebar"):
        if not df_filtered.empty:
            facts = []
            longest_msg_row = df_filtered.loc[df_filtered['Message_Length_Chars'].idxmax()]
            facts.append(f"Longest message: {longest_msg_row['Message_Length_Chars']} chars by {longest_msg_row['Author']} on {longest_msg_row['Date_Only'].strftime('%b %d, %Y')}!")
            max_emoji_msg_row = df_filtered.loc[df_filtered['Emoji_Count'].idxmax()]
            if max_emoji_msg_row['Emoji_Count'] > 0: facts.append(f"{max_emoji_msg_row['Author']} used {max_emoji_msg_row['Emoji_Count']} emojis in one message!")
            if not df_filtered.groupby('Date_Only')['Message'].count().empty:
                busiest_day_count = df_filtered.groupby('Date_Only')['Message'].count().max()
                busiest_day = df_filtered.groupby('Date_Only')['Message'].count().idxmax()
                facts.append(f"Busiest day: {busiest_day.strftime('%A, %b %d, %Y')} with {busiest_day_count} messages!")
            
            # Calculate Has_Love_Keyword on the fly for the fact if not present
            if 'Has_Love_Keyword' not in df_filtered.columns:
                df_filtered_for_fact = df_filtered.copy() # Avoid modifying df_filtered directly here
                df_filtered_for_fact['Has_Love_Keyword'] = df_filtered_for_fact['Message'].apply(lambda x: any(keyword.lower() in str(x).lower() for keyword in LOVE_KEYWORDS))
                total_love_msgs = df_filtered_for_fact['Has_Love_Keyword'].sum()
            else:
                total_love_msgs = df_filtered['Has_Love_Keyword'].sum()

            if total_love_msgs > 0: facts.append(f"Love expressed (keywords) in {total_love_msgs} messages!")

            if facts: st.sidebar.info(random.choice(facts))
            else: st.sidebar.warning("Could not generate a fun fact with current data filters.")
        else: st.sidebar.warning("Not enough data (after filters) for a fun fact!")
    
    st.sidebar.markdown("---"); st.sidebar.subheader("⏳ Nostalgia Corner")
    today_date_obj_sidebar = pd.to_datetime('today').date()
    # On This Day uses df_initial to show full history regardless of date filter
    on_this_day_messages = df_initial[
        (df_initial['Timestamp'].dt.month == today_date_obj_sidebar.month) &
        (df_initial['Timestamp'].dt.day == today_date_obj_sidebar.day) &
        (df_initial['Timestamp'].dt.year != today_date_obj_sidebar.year)
    ]
    if not on_this_day_messages.empty:
        st.sidebar.markdown(f"**On {today_date_obj_sidebar.strftime('%B %d')} in the past:**")
        for year_val_otd in sorted(on_this_day_messages['Year'].unique()): # Unique variable name
            year_messages_otd = on_this_day_messages[on_this_day_messages['Year'] == year_val_otd]
            st.sidebar.markdown(f"**In {year_val_otd}:**")
            for idx_otd, row_otd in year_messages_otd.head(2).iterrows(): # Unique variable names
                 st.sidebar.caption(f"_{row_otd['Author']}: \"{row_otd['Message'][:50]}...\"_")
            if len(year_messages_otd) > 2: st.sidebar.caption(f"...and {len(year_messages_otd)-2} more.")
    else: st.sidebar.markdown(f"No messages from {today_date_obj_sidebar.strftime('%B %d')} in previous years.")

    tab_names = [
        "📊 Overall Stats", "🕒 Temporal Analysis", "📜 Content Deep Dive", "💖 Sentiment & Love", 
        "✍️ Typing Styles (Evolution)", "🖼️ Media & Links", "🧠 Vocab Richness", 
        "🔤 Linguistic Style", "💬 Convo Dynamics", "🎉 Special Content"
    ]
    tabs = st.tabs(tab_names)

    # --- Tab 0: Overall Stats ---
    with tabs[0]:
        st.header("Overall Chat Statistics")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        total_messages = len(df_filtered); total_media = df_filtered['Is_Media'].sum()
        total_links = df_filtered['Is_Link'].sum(); total_words = df_filtered['Word_Count'].sum()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", f"{total_messages:,}"); col2.metric("Total Media", f"{total_media:,}")
        col3.metric("Total Links", f"{total_links:,}"); col4.metric("Total Words", f"{total_words:,}")

        st.subheader("Message Distribution")
        author_counts = df_filtered['Author'].value_counts()
        if not author_counts.empty:
            fig_msg_dist = px.pie(author_counts, values=author_counts.values, names=author_counts.index, title="Message Distribution", hole=0.3)
            fig_msg_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_msg_dist, use_container_width=True, key="pie_msg_dist_tab0")
            st.caption("Percentage of total messages sent by each person in the selected period.")

        st.subheader("Word Contribution")
        word_counts_author = df_filtered.groupby('Author')['Word_Count'].sum()
        if not word_counts_author.empty:
            fig_word_dist = px.pie(word_counts_author, values=word_counts_author.values, names=word_counts_author.index, title="Word Contribution", hole=0.3)
            fig_word_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_word_dist, use_container_width=True, key="pie_word_dist_tab0")
            st.caption("Percentage of total words written by each person in the selected period.")

        st.subheader("Average Message Length (Words)")
        avg_word_count = df_filtered.groupby('Author')['Word_Count'].mean().round(2)
        if not avg_word_count.empty: st.bar_chart(avg_word_count)

        if len(df_filtered) > 1: # For Longest Silence
            df_temp_silence = df_filtered.copy()
            df_temp_silence['Time_Diff_Overall'] = df_temp_silence['Timestamp'].diff()
            longest_silence = df_temp_silence['Time_Diff_Overall'].max()
            if pd.notna(longest_silence) and longest_silence > pd.Timedelta(seconds=0):
                silence_start_idx = df_temp_silence['Time_Diff_Overall'].idxmax()
                silence_start_time = df_temp_silence.loc[silence_start_idx - 1, 'Timestamp'] if silence_start_idx > 0 else "N/A"
                days = longest_silence.days; seconds = longest_silence.seconds
                hours = seconds // 3600; minutes = (seconds % 3600) // 60
                formatted_silence = f"{days}d {hours}h {minutes}m" if days > 0 else f"{hours}h {minutes}m"
                st.subheader("Longest Silence Streak 🤫")
                st.metric("Longest period without a message:", formatted_silence)
                if silence_start_time != "N/A": st.caption(f"Silence started after message on {silence_start_time.strftime('%Y-%m-%d %H:%M')}.")

    # --- Tab 1: Temporal Analysis ---
    with tabs[1]:
        st.header("When Do We Talk The Most?")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.subheader("Messages Over Time")
        messages_over_time = df_filtered.groupby('Date_Only')['Message'].count().reset_index() # Date_Only is already date object
        if not messages_over_time.empty:
            fig_timeline = px.line(messages_over_time, x='Date_Only', y='Message', title="Daily Message Volume")
            fig_timeline.update_xaxes(title_text='Date'); fig_timeline.update_yaxes(title_text='Number of Messages')
            st.plotly_chart(fig_timeline, use_container_width=True, key="line_msg_over_time_tab1")
            st.caption("Total number of messages exchanged each day over the chat history.")

        st.subheader("Activity by Hour of Day")
        hourly_activity = df_filtered.groupby(['Hour', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
        all_hours_df = pd.DataFrame({'Hour': range(24)})
        hourly_activity = pd.merge(all_hours_df, hourly_activity, on='Hour', how='left').fillna(0)
        authors_in_hourly = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in hourly_activity.columns]
        if authors_in_hourly and hourly_activity[authors_in_hourly].sum().sum() > 0:
            fig_hourly = px.bar(hourly_activity, x='Hour', y=authors_in_hourly, title="Hourly Chat Activity", barmode='group', labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_hourly, use_container_width=True, key="bar_hourly_activity_tab1")
            st.caption("Distribution of messages sent by each person across different hours of the day.")
        
        st.subheader("Activity by Day of Week")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_activity = df_filtered.groupby(['Day_Name', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
        daily_activity['Day_Name'] = pd.Categorical(daily_activity['Day_Name'], categories=days_order, ordered=True)
        daily_activity = daily_activity.sort_values('Day_Name')
        authors_in_daily = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in daily_activity.columns]
        if authors_in_daily and daily_activity[authors_in_daily].sum().sum() > 0:
            fig_daily = px.bar(daily_activity, x='Day_Name', y=authors_in_daily, title="Chat Activity by Day of the Week", barmode='group', labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_daily, use_container_width=True, key="bar_daily_activity_tab1")
            st.caption("Message activity broken down by the day of the week.")

        st.subheader("Monthly Message Contribution")
        monthly_activity = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
        monthly_activity.sort_values(['Year', 'Month_Num'], inplace=True)
        monthly_activity['Year-Month'] = monthly_activity['Year'].astype(str) + "-" + monthly_activity['Month']
        authors_in_monthly = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in monthly_activity.columns]
        if authors_in_monthly and monthly_activity[authors_in_monthly].sum().sum() > 0:
            fig_monthly = px.bar(monthly_activity, x='Year-Month', y=authors_in_monthly, title="Who Texts More Each Month?", barmode='group', labels={'value': 'Number of Messages', 'variable': 'Author'})
            st.plotly_chart(fig_monthly, use_container_width=True, key="bar_monthly_contrib_tab1")
            st.caption("Comparison of message counts per person for each month in the chat history.")
    # --- Tab 2: Content Deep Dive ---
    with tabs[2]:
        st.header("What Do We Talk About?")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
    
        df_text_only_cdd = df_filtered[~df_filtered['Is_Media']].copy()
        custom_stopwords_cdd_input = st.text_input("Add custom stopwords for Content Dive (comma-separated):", key="text_input_custom_stopwords_cdd_tab2")
        custom_sw_list_cdd = [sw.strip().lower() for sw in custom_stopwords_cdd_input.split(',') if sw.strip()]
        df_text_only_cdd['Cleaned_Message'] = df_text_only_cdd['Message'].apply(lambda x: get_cleaned_text(x, custom_stopwords=custom_sw_list_cdd))

        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.subheader(f"Word Cloud: {AUTHOR_1}")
            text_author1_wc = " ".join(df_text_only_cdd[df_text_only_cdd['Author'] == AUTHOR_1]['Cleaned_Message'])
            if text_author1_wc.strip():
                wordcloud1 = WordCloud(width=600, height=400, background_color='white', collocations=False).generate(text_author1_wc)
                fig_wc1, ax_wc1 = plt.subplots(); ax_wc1.imshow(wordcloud1, interpolation='bilinear'); ax_wc1.axis('off')
                st.pyplot(fig_wc1)
        with col_wc2:
            st.subheader(f"Word Cloud: {AUTHOR_2}")
            text_author2_wc = " ".join(df_text_only_cdd[df_text_only_cdd['Author'] == AUTHOR_2]['Cleaned_Message'])
            if text_author2_wc.strip():
                wordcloud2 = WordCloud(width=600, height=400, background_color='white', collocations=False).generate(text_author2_wc)
                fig_wc2, ax_wc2 = plt.subplots(); ax_wc2.imshow(wordcloud2, interpolation='bilinear'); ax_wc2.axis('off')
                st.pyplot(fig_wc2)

        st.subheader("Most Common Words (Cleaned)")
        num_common_words_cdd = st.slider("Number of common words:", 5, 30, 10, key="slider_common_words_cdd_tab2")
        all_cleaned_words_cdd = " ".join(df_text_only_cdd['Cleaned_Message']).split()
        if all_cleaned_words_cdd:
            common_words_overall_cdd = Counter(all_cleaned_words_cdd).most_common(num_common_words_cdd)
            df_common_overall_cdd_df = pd.DataFrame(common_words_overall_cdd, columns=['Word', 'Frequency'])
            fig_common_overall_cdd = px.bar(df_common_overall_cdd_df, x='Frequency', y='Word', orientation='h', title="Overall Most Common Words")
            st.plotly_chart(fig_common_overall_cdd, use_container_width=True, key="bar_common_overall_cdd_tab2")
            with st.expander("View Raw Data for Overall Common Words"):
                st.dataframe(df_common_overall_cdd_df)
        
        col_mcw1, col_mcw2 = st.columns(2)
        with col_mcw1:
            st.markdown(f"**{AUTHOR_1}'s Common Words**")
            words_auth1_cdd = " ".join(df_text_only_cdd[df_text_only_cdd['Author'] == AUTHOR_1]['Cleaned_Message']).split()
            if words_auth1_cdd:
                common_auth1_cdd_df = pd.DataFrame(Counter(words_auth1_cdd).most_common(num_common_words_cdd), columns=['Word', 'Frequency'])
                fig_common_auth1_cdd = px.bar(common_auth1_cdd_df, x='Frequency', y='Word', orientation='h')
                st.plotly_chart(fig_common_auth1_cdd, use_container_width=True, key="bar_common_auth1_cdd_tab2")
                with st.expander(f"View Raw Data for {AUTHOR_1}'s Common Words"):
                    st.dataframe(common_auth1_cdd_df)
        with col_mcw2:
            st.markdown(f"**{AUTHOR_2}'s Common Words**")
            words_auth2_cdd = " ".join(df_text_only_cdd[df_text_only_cdd['Author'] == AUTHOR_2]['Cleaned_Message']).split()
            if words_auth2_cdd:
                common_auth2_cdd_df = pd.DataFrame(Counter(words_auth2_cdd).most_common(num_common_words_cdd), columns=['Word', 'Frequency'])
                fig_common_auth2_cdd = px.bar(common_auth2_cdd_df, x='Frequency', y='Word', orientation='h')
                st.plotly_chart(fig_common_auth2_cdd, use_container_width=True, key="bar_common_auth2_cdd_tab2")
                with st.expander(f"View Raw Data for {AUTHOR_2}'s Common Words"):
                    st.dataframe(common_auth2_cdd_df)

        st.subheader("Emoji Analysis")
        col_ea1, col_ea2 = st.columns(2)
        with col_ea1:
            st.markdown(f"**{AUTHOR_1}'s Top Emojis**")
            emojis_auth1_str = "".join(df_author1[df_author1['Emoji_Count'] > 0]['Message'])
            if emojis_auth1_str:
                emoji_counts_auth1 = Counter(e_obj['emoji'] for e_obj in emoji.emoji_list(emojis_auth1_str)).most_common(10)
                if emoji_counts_auth1:
                    df_emoji_auth1 = pd.DataFrame(emoji_counts_auth1, columns=['Emoji', 'Frequency'])
                    fig_emoji_auth1 = px.bar(df_emoji_auth1, x='Frequency', y='Emoji', orientation='h')
                    st.plotly_chart(fig_emoji_auth1, use_container_width=True, key="bar_emoji_auth1_cdd_tab2")
        with col_ea2:
            st.markdown(f"**{AUTHOR_2}'s Top Emojis**")
            emojis_auth2_str = "".join(df_author2[df_author2['Emoji_Count'] > 0]['Message'])
            if emojis_auth2_str:
                emoji_counts_auth2 = Counter(e_obj['emoji'] for e_obj in emoji.emoji_list(emojis_auth2_str)).most_common(10)
                if emoji_counts_auth2:
                    df_emoji_auth2 = pd.DataFrame(emoji_counts_auth2, columns=['Emoji', 'Frequency'])
                    fig_emoji_auth2 = px.bar(df_emoji_auth2, x='Frequency', y='Emoji', orientation='h')
                    st.plotly_chart(fig_emoji_auth2, use_container_width=True, key="bar_emoji_auth2_cdd_tab2")

    # --- Tab 3: Sentiment & Love ---
    # (Assuming full code for this tab from your previous version is here, using df_filtered)
    # Ensure unique keys for any plotly charts, sliders, etc.
    # Add st.caption for date range at the top of the tab content.
    with tabs[3]:
        st.header("Sentiment & Expressions of Love")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        analyzer = SentimentIntensityAnalyzer()
        df_text_only_sentiment = df_filtered[~df_filtered['Is_Media']].copy()
        if not df_text_only_sentiment.empty:
            df_text_only_sentiment['Sentiment_Score'] = df_text_only_sentiment['Message'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
            def sentiment_category(score):
                if score > 0.05: return 'Positive'
                elif score < -0.05: return 'Negative'
                else: return 'Neutral'
            df_text_only_sentiment['Sentiment_Type'] = df_text_only_sentiment['Sentiment_Score'].apply(sentiment_category)

            st.subheader("Overall Sentiment Distribution")
            sentiment_counts = df_text_only_sentiment['Sentiment_Type'].value_counts()
            if not sentiment_counts.empty:
                fig_sentiment_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Overall Sentiment of Messages", hole=0.3)
                st.plotly_chart(fig_sentiment_pie, use_container_width=True, key="pie_sentiment_overall_tab3")
            
            st.subheader("Sentiment by Author")
            sentiment_by_author = df_text_only_sentiment.groupby('Author')['Sentiment_Score'].mean().reset_index()
            if not sentiment_by_author.empty:
                fig_sentiment_author = px.bar(sentiment_by_author, x='Author', y='Sentiment_Score', color='Author', title="Average Sentiment Score per Author")
                st.plotly_chart(fig_sentiment_author, use_container_width=True, key="bar_sentiment_author_tab3")

            st.subheader("Sentiment Over Time")
            sentiment_over_time = df_text_only_sentiment.groupby('Date_Only')['Sentiment_Score'].mean().reset_index() # Use Date_Only
            if not sentiment_over_time.empty:
                fig_sentiment_timeline = px.line(sentiment_over_time, x='Date_Only', y='Sentiment_Score', title="Average Daily Sentiment") # Use Date_Only
                fig_sentiment_timeline.add_hline(y=0, line_dash="dot", annotation_text="Neutral")
                st.plotly_chart(fig_sentiment_timeline, use_container_width=True, key="line_sentiment_time_tab3")

        st.subheader("Expressions of Love ❤️")
        if 'Has_Love_Keyword' not in df_filtered.columns: # Ensure column exists
                df_filtered['Has_Love_Keyword'] = df_filtered['Message'].apply(
                lambda x: any(keyword.lower() in str(x).lower() for keyword in LOVE_KEYWORDS))
        love_expressions_count = df_filtered.groupby('Author')['Has_Love_Keyword'].sum()
        if not love_expressions_count.empty and love_expressions_count.sum() > 0:
            fig_love_tab3 = px.bar(love_expressions_count, x=love_expressions_count.index, y=love_expressions_count.values,
                            color=love_expressions_count.index, title="Who Expresses 'Love' More (Keyword Based)?",
                            labels={'y': 'Messages with Love Keywords', 'index': 'Author'})
            st.plotly_chart(fig_love_tab3, use_container_width=True, key="bar_love_expressions_tab3")
        else: st.write("No 'love' keywords found or data unavailable.")
        st.markdown(f"**Keywords used (sample):** `{', '.join(LOVE_KEYWORDS[:7])}...` Customize this list in the script!")


    # --- Tab 4: Typing Styles (Evolution) ---
    # (Assuming full code for this tab from your previous version is here, using df_filtered)
    with tabs[4]:
        st.header("Evolution of Typing Styles")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.subheader("Average Message Length (Words) Over Time")
        avg_len_words_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Word_Count'].mean().unstack(fill_value=0).reset_index()
        avg_len_words_time.sort_values(['Year', 'Month_Num'], inplace=True)
        avg_len_words_time['Year-Month'] = avg_len_words_time['Year'].astype(str) + "-" + avg_len_words_time['Month']
        authors_in_avglen_words = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in avg_len_words_time.columns]
        if authors_in_avglen_words and avg_len_words_time[authors_in_avglen_words].sum().sum() > 0:
            fig_avg_len_words_time = px.line(avg_len_words_time, x='Year-Month', y=authors_in_avglen_words,
                                    title="Average Message Length (Words) per Month",
                                    labels={'value': 'Avg Words/Message', 'variable': 'Author'})
            st.plotly_chart(fig_avg_len_words_time, use_container_width=True, key="line_avg_msg_len_words_tab4")

        st.subheader("Emoji Usage Over Time (Total Emojis Sent)")
        emoji_usage_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Emoji_Count'].sum().unstack(fill_value=0).reset_index()
        emoji_usage_time.sort_values(['Year', 'Month_Num'], inplace=True)
        emoji_usage_time['Year-Month'] = emoji_usage_time['Year'].astype(str) + "-" + emoji_usage_time['Month']
        authors_in_emoji_time = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in emoji_usage_time.columns]
        if authors_in_emoji_time and emoji_usage_time[authors_in_emoji_time].sum().sum() > 0:
            fig_emoji_time = px.line(emoji_usage_time, x='Year-Month', y=authors_in_emoji_time,
                                title="Total Emojis Used per Month",
                                labels={'value': 'Total Emojis', 'variable': 'Author'})
            st.plotly_chart(fig_emoji_time, use_container_width=True, key="line_emoji_usage_time_tab4")
        # ... (Avg Message Length (Words) Over Time with unique key)
        # ... (Emoji Usage Over Time with unique key)

    # --- Tab 5: Media & Links ---
    # (Assuming full code for this tab from your previous version is here, using df_filtered)
    with tabs[5]:
        st.header("Media and Link Sharing")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.subheader("Media Shared Distribution")
        media_counts_author = df_filtered.groupby('Author')['Is_Media'].sum()
        if not media_counts_author.empty and media_counts_author.sum() > 0:
            fig_media_dist = px.pie(media_counts_author, values=media_counts_author.values, names=media_counts_author.index, title="Who Shared More Media?", hole=0.3)
            fig_media_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_media_dist, use_container_width=True, key="pie_media_dist_tab5")

        st.subheader("Links Shared Distribution")
        link_counts_author = df_filtered.groupby('Author')['Is_Link'].sum()
        if not link_counts_author.empty and link_counts_author.sum() > 0:
            fig_link_dist = px.pie(link_counts_author, values=link_counts_author.values, names=link_counts_author.index, title="Who Shared More Links?", hole=0.3)
            fig_link_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_link_dist, use_container_width=True, key="pie_link_dist_tab5")
        
        st.subheader("Hourly Media Sharing")
        media_by_hour_tab5 = df_filtered[df_filtered['Is_Media']].groupby(['Hour', 'Author'])['Message'].count().unstack(fill_value=0).reset_index()
        if not media_by_hour_tab5.empty:
            all_hours_df_media_tab5 = pd.DataFrame({'Hour': range(24)})
            media_by_hour_tab5 = pd.merge(all_hours_df_media_tab5, media_by_hour_tab5, on='Hour', how='left').fillna(0)
            authors_in_media_hourly = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in media_by_hour_tab5.columns]
            if authors_in_media_hourly and media_by_hour_tab5[authors_in_media_hourly].sum().sum() > 0:
                fig_media_hourly_tab5 = px.bar(media_by_hour_tab5, x='Hour', y=authors_in_media_hourly, title="Hourly Media Sharing Activity", barmode='group', labels={'value': 'Number of Media Messages', 'variable': 'Author'})
                st.plotly_chart(fig_media_hourly_tab5, use_container_width=True, key="bar_media_hourly_tab5")
        # ... (Media & Links pie charts and hourly media bar chart with unique keys)

    # --- Tab 6: Vocabulary Richness (Overall) ---
    # (Assuming full code for this tab from your previous version is here, using df_filtered)
    with tabs[6]:
        st.header("🧠 Overall Vocabulary Richness")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        df_text_vocab = df_filtered[~df_filtered['Is_Media']].copy()
        df_text_vocab['Cleaned_Vocab'] = df_text_vocab['Message'].apply(lambda x: get_cleaned_text(x))
        vocab_stats_list = []
        for author_v in [AUTHOR_1, AUTHOR_2]:
            author_texts_v = df_text_vocab[df_text_vocab['Author'] == author_v]['Cleaned_Vocab'].tolist()
            ttr_v = calculate_ttr(author_texts_v)
            all_words_v = " ".join(author_texts_v).split()
            vocab_stats_list.append({
                'Author': author_v, 'Total Words (Cleaned)': len(all_words_v),
                'Unique Words (Cleaned)': len(set(all_words_v)), 'Overall TTR (%)': round(ttr_v, 2)
            })
        if vocab_stats_list: st.dataframe(pd.DataFrame(vocab_stats_list).set_index('Author'))
        
        st.subheader("Top Unique Words (Cleaned)")
        top_n_vocab = st.slider("Select N for top unique words:", 5, 50, 15, key="slider_top_n_vocab_tab6")
        col_tun1, col_tun2 = st.columns(2)
        with col_tun1:
            st.markdown(f"**{AUTHOR_1}'s Top {top_n_vocab} Unique Words**")
            words_auth1_vocab = " ".join(df_text_vocab[df_text_vocab['Author'] == AUTHOR_1]['Cleaned_Vocab']).split()
            if words_auth1_vocab:
                st.dataframe(pd.DataFrame(Counter(words_auth1_vocab).most_common(top_n_vocab), columns=['Word', 'Frequency']))
        with col_tun2:
            st.markdown(f"**{AUTHOR_2}'s Top {top_n_vocab} Unique Words**")
            words_auth2_vocab = " ".join(df_text_vocab[df_text_vocab['Author'] == AUTHOR_2]['Cleaned_Vocab']).split()
            if words_auth2_vocab:
                st.dataframe(pd.DataFrame(Counter(words_auth2_vocab).most_common(top_n_vocab), columns=['Word', 'Frequency']))

        # ... (Vocabulary Richness dataframe and Top Unique Words dataframes)

    # --- Tab 7: Linguistic Style (Detailed) ---
    # (Assuming full code for this tab from your previous version is here, using df_filtered and df_author1/2)
    with tabs[7]:
        st.header("🔤 Detailed Linguistic Styles")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.subheader("Typing Style Indicators (Overall Totals)")
        col_ls_m1, col_ls_m2 = st.columns(2)
        with col_ls_m1:
            st.metric(f"Total Exclamations by {AUTHOR_1}", df_author1['Exclamation_Count'].sum())
            st.metric(f"Total Questions by {AUTHOR_1}", df_author1['Question_Count'].sum())
            st.metric(f"Total Elongations by {AUTHOR_1}", df_author1['Elongations'].sum())
            st.metric(f"Total CAPS Msgs by {AUTHOR_1}", df_author1['Is_Caps'].sum())
        with col_ls_m2:
            st.metric(f"Total Exclamations by {AUTHOR_2}", df_author2['Exclamation_Count'].sum())
            st.metric(f"Total Questions by {AUTHOR_2}", df_author2['Question_Count'].sum())
            st.metric(f"Total Elongations by {AUTHOR_2}", df_author2['Elongations'].sum())
            st.metric(f"Total CAPS Msgs by {AUTHOR_2}", df_author2['Is_Caps'].sum())

        st.subheader("Message Length Evolution (Characters)")
        avg_len_chars_time = df_filtered.groupby(['Year', 'Month_Num', 'Month', 'Author'])['Message_Length_Chars'].mean().unstack(fill_value=0).reset_index()
        avg_len_chars_time.sort_values(['Year', 'Month_Num'], inplace=True)
        avg_len_chars_time['Year-Month'] = avg_len_chars_time['Year'].astype(str) + "-" + avg_len_chars_time['Month']
        authors_in_avglen_chars = [auth for auth in [AUTHOR_1, AUTHOR_2] if auth in avg_len_chars_time.columns]
        if authors_in_avglen_chars and avg_len_chars_time[authors_in_avglen_chars].sum().sum() > 0:
            fig_avg_len_chars_time = px.line(avg_len_chars_time, x='Year-Month', y=authors_in_avglen_chars, title="Avg Message Length (Characters) per Month", labels={'value': 'Avg Chars/Message', 'variable': 'Author'})
            st.plotly_chart(fig_avg_len_chars_time, use_container_width=True, key="line_avg_msg_len_chars_tab7")

        st.subheader("Vocabulary Richness (TTR) per Month")
        df_text_ttr_monthly = df_filtered[~df_filtered['Is_Media']].copy()
        df_text_ttr_monthly['Cleaned_TTR'] = df_text_ttr_monthly['Message'].apply(lambda x: get_cleaned_text(x))
        ttr_monthly_data = []
        for author_ttr_m in [AUTHOR_1, AUTHOR_2]:
            author_df_ttr_m = df_text_ttr_monthly[df_text_ttr_monthly['Author'] == author_ttr_m]
            for (year_ttr, month_name_ttr, month_num_ttr), group_ttr in author_df_ttr_m.groupby(['Year', 'Month', 'Month_Num']):
                ttr_val = calculate_ttr(group_ttr['Cleaned_TTR'].tolist())
                ttr_monthly_data.append({'Year-Month': f"{year_ttr}-{month_name_ttr}", 'Author': author_ttr_m, 'TTR': ttr_val, 'Year': year_ttr, 'Month_Num': month_num_ttr})
        if ttr_monthly_data:
            df_ttr_plot = pd.DataFrame(ttr_monthly_data).sort_values(['Year', 'Month_Num'])
            if not df_ttr_plot.empty:
                fig_ttr_monthly = px.line(df_ttr_plot, x='Year-Month', y='TTR', color='Author', title="Monthly TTR (%)")
                st.plotly_chart(fig_ttr_monthly, use_container_width=True, key="line_ttr_monthly_tab7")

        st.subheader("Language Use (Detected)")
        lang_counts_author_tab7 = df_filtered.groupby(['Author', 'Language'])['Message'].count().unstack(fill_value=0)
        if not lang_counts_author_tab7.empty:
            st.dataframe(lang_counts_author_tab7) # Display raw counts
            all_langs_sum_tab7 = lang_counts_author_tab7.sum().sort_values(ascending=False)
            common_langs_plot_tab7 = [lang for lang in all_langs_sum_tab7.index if lang not in ['unknown', 'error_langdetect', 'error_other_lang']][:5]
            if common_langs_plot_tab7:
                lang_melted_tab7 = lang_counts_author_tab7[common_langs_plot_tab7].reset_index().melt(id_vars='Author', value_vars=common_langs_plot_tab7, var_name='Language', value_name='Count')
                if not lang_melted_tab7.empty and lang_melted_tab7['Count'].sum() > 0:
                    fig_lang_tab7 = px.bar(lang_melted_tab7, x='Author', y='Count', color='Language', barmode='group', title="Top Detected Languages")
                    st.plotly_chart(fig_lang_tab7, use_container_width=True, key="bar_lang_use_tab7")

        st.subheader("Slang / Nickname Usage")
        slang_data_tab7 = []
        for author_slang, slang_list in SLANG_NICKNAMES.items():
            if author_slang == 'common': continue
            for term in slang_list:
                count = df_filtered[(df_filtered['Author'] == author_slang) & (df_filtered['Message'].str.contains(term, case=False, na=False))].shape[0]
                if count > 0: slang_data_tab7.append({'Author': author_slang, 'Term': term, 'Count': count})
        for term_common in SLANG_NICKNAMES.get('common', []):
            for author_common_slang in [AUTHOR_1, AUTHOR_2]:
                count_common = df_filtered[(df_filtered['Author'] == author_common_slang) & (df_filtered['Message'].str.contains(term_common, case=False, na=False))].shape[0]
                if count_common > 0: slang_data_tab7.append({'Author': author_common_slang, 'Term': f"{term_common} (common)", 'Count': count_common})
        if slang_data_tab7:
            df_slang_tab7 = pd.DataFrame(slang_data_tab7)
            if not df_slang_tab7.empty:
                df_slang_tab7 = df_slang_tab7[df_slang_tab7['Count'] > 0].sort_values('Count', ascending=False)
                if not df_slang_tab7.empty:
                    fig_slang_tab7 = px.bar(df_slang_tab7, x='Term', y='Count', color='Author', barmode='group', title="Slang/Nickname Usage")
                    st.plotly_chart(fig_slang_tab7, use_container_width=True, key="bar_slang_usage_tab7")
        # ... (Metrics for Exclamation_Count etc. - no keys needed)
        # ... (Message Length (Chars) Evolution plot with unique key)
        # ... (Monthly TTR plot with unique key)
        # ... (Language Use plot with unique key)
        # ... (Slang/Nickname Usage plot with unique key)

    # --- Tab 8: Conversation Dynamics ---
    with tabs[8]:
        st.header("💬 Conversation Dynamics")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.caption(f"Conversation initiator threshold: {silence_threshold_minutes_input} minutes of silence.")
        st.subheader("Average Response Time (Seconds)")
        valid_response_times = df_filtered[df_filtered['Response_Time_Seconds'].notna() & (df_filtered['Response_Time_Seconds'] > 0)]
        if not valid_response_times.empty:
            avg_response_time_author = valid_response_times.groupby('Author')['Response_Time_Seconds'].mean().round(2)
            if not avg_response_time_author.empty:
                st.bar_chart(avg_response_time_author)
                st.caption("Average time taken by each person to reply to the other's message.")
        else: st.write("Not enough back-and-forth messages to calculate response times.")

        st.subheader("Who Initiates Conversations More?")
        initiator_counts = df_filtered[df_filtered['Is_Initiator']].groupby('Author')['Message'].count()
        if not initiator_counts.empty:
            fig_initiator = px.pie(initiator_counts, values=initiator_counts.values, names=initiator_counts.index, title="Conversation Initiators (>30min silence)", hole=0.3)
            fig_initiator.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_initiator, use_container_width=True, key="pie_initiators_tab8")
            st.caption("Identifies who sends the first message after a period of more than 30 minutes of silence.")

        st.subheader("Consecutive Messages (Streaks)")
        max_consecutive_author = df_filtered.groupby('Author')['Consecutive_Count'].max()
        avg_consecutive_author = df_filtered[df_filtered['Consecutive_Count'] > 1].groupby('Author')['Consecutive_Count'].mean().round(2)
        col_streak1, col_streak2 = st.columns(2)
        with col_streak1:
            st.metric(f"Longest Streak by {AUTHOR_1}", max_consecutive_author.get(AUTHOR_1, 0))
            st.metric(f"Avg. Streak Length by {AUTHOR_1}", avg_consecutive_author.get(AUTHOR_1, 0.0))
        with col_streak2:
            st.metric(f"Longest Streak by {AUTHOR_2}", max_consecutive_author.get(AUTHOR_2, 0))
            st.metric(f"Avg. Streak Length by {AUTHOR_2}", avg_consecutive_author.get(AUTHOR_2, 0.0))
        st.caption("A streak is a series of messages sent by one person before the other replies. Average streak length considers streaks of more than 1 message.")
    # ... (Response Time bar chart - no key needed)
        # ... (Initiators pie chart with unique key)
        # ... (Consecutive Messages metrics - no keys needed)

    # --- Tab 9: Special Content --- (Renamed from Special Content & Trends for clarity)
    with tabs[9]:
        st.header("🎉 Special Content")
        st.caption(f"Displaying data from: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")
        st.subheader("Expressions of Love ❤️ (Keyword Based)")
        if 'Has_Love_Keyword' not in df_filtered.columns: # Ensure column exists
                df_filtered['Has_Love_Keyword'] = df_filtered['Message'].apply(
                lambda x: any(keyword.lower() in str(x).lower() for keyword in LOVE_KEYWORDS))
        love_expressions_count_tab9 = df_filtered.groupby('Author')['Has_Love_Keyword'].sum()
        if not love_expressions_count_tab9.empty and love_expressions_count_tab9.sum() > 0:
            fig_love_tab9 = px.bar(love_expressions_count_tab9, x=love_expressions_count_tab9.index,
                                    y=love_expressions_count_tab9.values, color=love_expressions_count_tab9.index,
                                    title="Love Expressions (Keywords) - Special Content",
                                    labels={'y': 'Love Keyword Messages', 'index': 'Author'})
            st.plotly_chart(fig_love_tab9, use_container_width=True, key="bar_love_expressions_tab9")
        else: st.write("No 'love' keyword messages found.")
        st.markdown(f"**Keywords used (sample):** `{', '.join(LOVE_KEYWORDS[:7])}...`")

        st.subheader("Peak Messaging Days")
        daily_msg_counts = df_filtered.groupby('Date_Only')['Message'].count().reset_index() # Use Date_Only
        daily_msg_counts.rename(columns={'Message': 'Total_Messages'}, inplace=True)
        daily_msg_counts.sort_values('Total_Messages', ascending=False, inplace=True)
        num_peak_days = st.slider("Number of Peak Days to Show:", 1, min(20, len(daily_msg_counts)), 5, key="slider_peak_days_tab9")
        
        if not daily_msg_counts.empty:
            peak_days_df = daily_msg_counts.head(num_peak_days)
            st.dataframe(peak_days_df)
            st.caption("Days with the highest message volume. Explore them for special occasions!")

            st.subheader("Context from Peak Days")
            for index, peak_row in peak_days_df.iterrows():
                peak_date_obj_context = peak_row['Date_Only'] # This should be a datetime.date object
                st.markdown(f"**On {peak_date_obj_context.strftime('%Y-%m-%d')} (Total: {peak_row['Total_Messages']} messages):**")
                
                messages_on_peak_date = df_filtered[df_filtered['Date_Only'] == peak_date_obj_context]

                if not messages_on_peak_date.empty:
                    st.markdown("*Sample Start of Day:*")
                    for idx_msg, msg_row in messages_on_peak_date.head(2).iterrows():
                        st.text(f"  {msg_row['Timestamp'].strftime('%H:%M')} - {msg_row['Author']}: {msg_row['Message'][:70]}...")
                    if len(messages_on_peak_date) > 5:
                        middle_idx = len(messages_on_peak_date) // 2
                        mid_msg_row = messages_on_peak_date.iloc[middle_idx]
                        st.markdown("*Sample Mid-Day:*")
                        st.text(f"  {mid_msg_row['Timestamp'].strftime('%H:%M')} - {mid_msg_row['Author']}: {mid_msg_row['Message'][:70]}...")
                    if len(messages_on_peak_date) > 2:
                        st.markdown("*Sample End of Day:*")
                        for idx_msg, msg_row in messages_on_peak_date.tail(2).iterrows():
                            st.text(f"  {msg_row['Timestamp'].strftime('%H:%M')} - {msg_row['Author']}: {msg_row['Message'][:70]}...")
                else:
                    st.write(f"Could not retrieve messages for {peak_date_obj_context.strftime('%Y-%m-%d')}.")
                
                # Greeting/Celebration term detection within the loop for each peak day
                greeting_terms_on_this_peak_day = messages_on_peak_date.copy()
                greeting_terms_on_this_peak_day['Found_Greetings'] = greeting_terms_on_this_peak_day['Message'].apply(
                    lambda m: ", ".join([term for term in GREETING_CELEBRATION_TERMS if term.lower() in str(m).lower()]) or None
                )
                greeting_msgs_on_this_peak = greeting_terms_on_this_peak_day[greeting_terms_on_this_peak_day['Found_Greetings'].notna()]
                if not greeting_msgs_on_this_peak.empty:
                    with st.expander(f"Show Greeting/Celebration Messages for {peak_date_obj_context.strftime('%Y-%m-%d')}"):
                        st.dataframe(greeting_msgs_on_this_peak[['Timestamp', 'Author', 'Message', 'Found_Greetings']])
                st.markdown("---")
        # ... (Love Expressions bar chart with unique key, using df_filtered)
        # ... (Peak Messaging Days slider with unique key and context display with expanders, using df_filtered)

else:
    if uploaded_file and (df_initial is None or df_initial.empty):
         st.warning("Uploaded CSV processed, but no valid data found for authors or general content.")
   
    
    elif not uploaded_file:
        st.info("Awaiting your WhatsApp chat CSV file to begin the journey... 😊")


st.sidebar.markdown("---")
st.sidebar.markdown("Crafted with data and love!")