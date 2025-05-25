import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime, timedelta
import altair as alt
from load_data import process_json_data
from fetchdata import get_data
import csv, requests
from collections import defaultdict
from model import make_pred
from model import YayOrNay

# Set page configuration
st.set_page_config(
    page_title="Reddit Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)



subreddit_name = st.text_input("Enter Your Subreddit Name")
submit = st.button("Submit")

data, post_info = None , None 
if submit and subreddit_name:
    # Load data
    try:
        print("lkjlfkjd")
        get_data(limit=1, subreddit_name=subreddit_name)
    except Exception as e:
        print(e, "data could not be fetched")
    data, post_info = process_json_data("reddit_data.json")

# App title and description
st.title("Reddit Sentiment Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

# Date range filter
min_date = data['date'].min().date()
max_date = data['date'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    data_filtered = data[(data['date'].dt.date >= start_date) & (data['date'].dt.date <= end_date)]
else:
    data_filtered = data.copy()

# Sentiment filter
sentiment_options = ['All'] + list(data['sentiment'].unique())
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiment_options)
if selected_sentiment != 'All':
    data_filtered = data_filtered[data_filtered['sentiment'] == selected_sentiment]

# Topic filter
topic_options = ['All'] + list(data['topic'].unique())
selected_topic = st.sidebar.selectbox("Topic", topic_options)
if selected_topic != 'All':
    data_filtered = data_filtered[data_filtered['topic'] == selected_topic]

# User karma slider
min_karma, max_karma = int(data['user_karma'].min()), int(data['user_karma'].max())
karma_range = st.sidebar.slider(
    "User Karma Range",
    min_karma, max_karma,
    (min_karma, max_karma)
)
data_filtered = data_filtered[
    (data_filtered['user_karma'] >= karma_range[0]) & 
    (data_filtered['user_karma'] <= karma_range[1])
]

# Advanced Options
st.sidebar.header("Advanced Options")

# Visualization settings
chart_height = st.sidebar.slider("Chart Height", 300, 800, 400)
color_theme = st.sidebar.selectbox("Color Theme", ["blues", "viridis", "plasma", "inferno", "magma", "cividis"])

# Create layout with columns for dashboard
col1, col2 = st.columns(2)

# KPI metrics at the top
st.header("Key Metrics")
metrics_cols = st.columns(4)

with metrics_cols[0]:
    st.metric(
        "Total Comments", 
        f"{len(data_filtered):,}",
        f"{len(data_filtered) - len(data)}"
    )
    
with metrics_cols[1]:
    positive_pct = (data_filtered['sentiment'] == 'Positive').mean() * 100
    overall_positive_pct = (data['sentiment'] == 'Positive').mean() * 100
    delta = positive_pct - overall_positive_pct
    st.metric(
        "Positive Sentiment", 
        f"{positive_pct:.1f}%",
        f"{delta:.1f}%"
    )
    
with metrics_cols[2]:
    neutral_pct = (data_filtered['sentiment'] == 'Neutral').mean() * 100
    overall_neutral_pct = (data['sentiment'] == 'Neutral').mean() * 100
    delta = neutral_pct - overall_neutral_pct
    st.metric(
        "Neutral Sentiment", 
        f"{neutral_pct:.1f}%",
        f"{delta:.1f}%"
    )
    
with metrics_cols[3]:
    negative_pct = (data_filtered['sentiment'] == 'Negative').mean() * 100
    overall_negative_pct = (data['sentiment'] == 'Negative').mean() * 100
    delta = negative_pct - overall_negative_pct
    st.metric(
        "Negative Sentiment", 
        f"{negative_pct:.1f}%",
        f"{delta:.1f}%"
    )

# Visualizations


# Word Cloud
st.header("Word Cloud by Sentiment")
word_cloud_cols = st.columns(3)

# Function to create word cloud
def create_word_cloud(data, sentiment):
    filtered_data = data[data['sentiment'] == sentiment]
    
    # Extract all keywords and join
    all_keywords = []
    for keywords_list in filtered_data['keywords']:
        all_keywords.extend(keywords_list)
    
    # Create word frequency dictionary
    word_freq = {}
    for word in all_keywords:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
    # Color maps for different sentiments
    color_maps = {
        'Positive': 'Greens',
        'Neutral': 'YlOrBr',
        'Negative': 'Reds'
    }
    
    # Generate word cloud
    if word_freq:
        wc = WordCloud(
            background_color='white',
            max_words=100,
            colormap=color_maps.get(sentiment, 'viridis'),
            width=800,
            height=400
        ).generate_from_frequencies(word_freq)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        ax.axis('off')
        return fig

# Display word clouds for each sentiment
with word_cloud_cols[0]:
    st.subheader("Positive Sentiment")
    wc_positive = create_word_cloud(data_filtered, 'Positive')
    st.pyplot(wc_positive)

with word_cloud_cols[1]:
    st.subheader("Neutral Sentiment")
    wc_neutral = create_word_cloud(data_filtered, 'Neutral')
    st.pyplot(wc_neutral)

with word_cloud_cols[2]:
    st.subheader("Negative Sentiment")
    wc_negative = create_word_cloud(data_filtered, 'Negative')
    st.pyplot(wc_negative)





# Sample comments section
st.header("Top Comments")
num_samples = st.slider("Number of top comments to display", 1, 20, 5)

# Sort by upvotes
sample_comments = data_filtered.sort_values('upvotes', ascending=False).head(num_samples)

for i, row in enumerate(sample_comments.iterrows()):
    comment = row[1]
    
    # Create a card-like container for each comment
    comment_container = st.container()
    with comment_container:
        st.markdown(f"""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
            <p><strong>u/{comment['author']}</strong> · {comment['date'].strftime('%Y-%m-%d %H:%M')} · {comment['upvotes']} upvotes</p>
            <p>{comment['text']}</p>
            <p>
                <span style="color:{'#28a745' if comment['sentiment'] == 'Positive' else '#ffc107' if comment['sentiment'] == 'Neutral' else '#dc3545'}">
                    {comment['sentiment']} (Score: {comment['sentiment_score']:.2f})
                </span> · 
                Topic: {comment['topic']} · 
                User Karma: {comment['user_karma']} · 
                Account Age: {comment['user_age_days']} days
            </p>
        </div>
        """, unsafe_allow_html=True)

# Data summary section
st.header("Data Summary")
show_summary = st.checkbox("Show data summary statistics")

if show_summary:
    st.write("### Filtered Data Summary")
    summary_df = data_filtered[['sentiment_score', 'upvotes', 'user_karma', 'user_age_days']].describe().T
    st.dataframe(summary_df)
    
    st.write("### Data Sample")
    st.dataframe(data_filtered[['author', 'text', 'sentiment', 'upvotes', 'topic']].head(10))


# Footer
st.markdown("""
---
<p style="text-align:center">Reddit Sentiment Analysis Dashboard | Created with Streamlit</p>
""", unsafe_allow_html=True)
