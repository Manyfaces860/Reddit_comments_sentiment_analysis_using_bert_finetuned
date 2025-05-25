import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import json
from datetime import datetime, timedelta
from model import make_pred
from model import YayOrNay
# from model import make_pred, YayOrNay, Custom, evaluate_model

# First ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    
def clean_text(text):
    """Clean and preprocess text for topic extraction"""
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove Reddit-specific formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    # Remove image links and other formatting
    text = re.sub(r'!\[.*?\]', '', text)
    text = re.sub(r'emote\|t5_\w+\|\d+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    return text.lower().strip()

def extract_topics(df, post_data, num_topics=5):
    """
    Extract topics from Reddit post and comments using TF-IDF and NLP techniques
    Works for any text category, not just cricket
    
    Args:
        df: DataFrame containing comments
        post_data: Dictionary with post information
        num_topics: Number of topics to extract
    
    Returns:
        Updated DataFrame with dynamically extracted topics
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get subreddit to help with context
    subreddit = post_data.get('subreddit', '').lower()
    
    # Combine post title, selftext, and comments for processing
    all_text = post_data['title'] + ' ' + post_data['selftext'] + ' '
    all_text += ' '.join(df['text'].tolist())
    
    # Clean the text
    cleaned_text = clean_text(all_text)
    
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Add general stopwords that don't contribute to topics
    general_stopwords = [
        'just', 'like', 'get', 'got', 'really', 'even', 'still', 'back',
        'make', 'see', 'think', 'going', 'didnt', 'doesnt', 'dont', 'cant',
        'day', 'guys', 'man', 'time', 'yeah', 'say', 'said', 'would', 'could',
        'should', 'way', 'thing', 'know', 'also', 'much', 'many', 'well', 'actually',
        'basically', 'look', 'looks', 'looking', 'lot', 'gonna', 'wanna', 'need',
        'maybe', 'sure', 'right', 'want', 'people', 'person', 'ever', 'one', 'two',
        'three', 'first', 'second', 'new', 'old', 'yes', 'no', 'good', 'bad'
    ]
    
    stop_words.update(general_stopwords)
    
    # Remove stopwords and lemmatize
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 2]
    
    # Get most common words - simple topic extraction method
    word_counts = Counter(filtered_tokens)
    common_words = [word for word, count in word_counts.most_common(num_topics*3) if count > 1]
    
    # Find significant bigrams (word pairs that occur together)
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(filtered_tokens)
    finder.apply_freq_filter(2)  # Only consider bigrams that appear at least twice
    significant_bigrams = finder.nbest(bigram_measures.pmi, 10)
    bigram_topics = [f"{w1} {w2}" for w1, w2 in significant_bigrams]
    
    # Process individual comments for vectorization
    comment_texts = [clean_text(text) for text in df['text']]
    
    # Combine with post title and selftext
    all_documents = [post_data['title'] + " " + post_data['selftext']] + comment_texts
    
    # Use TF-IDF to get more sophisticated topics
    tfidf_topics = []
    if len(" ".join(all_documents).split()) > 20:
        try:
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=list(stop_words),
                ngram_range=(1, 2)  # Include both single words and bigrams
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get highest TF-IDF scores by summing across all documents
            tfidf_sums = tfidf_matrix.sum(axis=0).A1
            top_indices = tfidf_sums.argsort()[-num_topics*2:][::-1]
            tfidf_topics = [feature_names[i] for i in top_indices]
            
            # Try LDA for topic modeling when we have enough data
            if len(all_documents) > 5:
                lda = LatentDirichletAllocation(
                    n_components=min(3, len(all_documents)-1),
                    random_state=42
                )
                lda.fit(tfidf_matrix)
                
                # Get top words for each topic
                lda_topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_word_indices = topic.argsort()[:-5:-1]  # Get 5 most important words
                    topic_words = [feature_names[i] for i in top_word_indices]
                    lda_topics.append(" ".join(topic_words))
                    
                tfidf_topics.extend(lda_topics)
        except Exception as e:
            print(f"TF-IDF processing error: {e}")
            # Fallback to simple word counting
            tfidf_topics = common_words
    else:
        tfidf_topics = common_words
    
    # Get context from subreddit and title
    title_words = clean_text(post_data['title']).split()
    title_cleaned = [lemmatizer.lemmatize(word) for word in title_words if word not in stop_words and len(word) > 2]
    
    # Extract entity/subject from title (usually the first few nouns)
    subject_candidates = title_cleaned[:3]
    
    # Create contextual topics based on subreddit and title
    contextual_topics = []
    
    # Add subreddit as a context if it's not too generic
    if subreddit and subreddit not in ['all', 'popular', 'news', 'askreddit']:
        # Convert subreddit name to a readable format
        readable_subreddit = ' '.join(word.capitalize() for word in re.findall(r'[A-Za-z]+', subreddit))
        if readable_subreddit:
            contextual_topics.append(readable_subreddit)
    
    # Look for specific patterns in the title that indicate topics
    title_lower = post_data['title'].lower()
    
    # Question patterns
    if title_lower.startswith(('what', 'why', 'how', 'when', 'who', 'is', 'are', 'can', 'should', 'do')) or '?' in title_lower:
        contextual_topics.append('Question')
        
        # Specific question types
        if any(word in title_lower for word in ['opinion', 'think', 'believe', 'thoughts']):
            contextual_topics.append('Opinion Poll')
    
    # Announcement patterns
    if any(word in title_lower for word in ['announcement', 'update', 'news', 'breaking']):
        contextual_topics.append('Announcement')
    
    # Discussion patterns
    if any(word in title_lower for word in ['discussion', 'thread', 'debate', 'talk']):
        contextual_topics.append('Discussion')
        
    # Humor patterns
    if any(word in title_lower for word in ['meme', 'joke', 'funny', 'humor', 'lol', 'haha']):
        contextual_topics.append('Humor')
        
    # Help/Advice patterns
    if any(word in title_lower for word in ['help', 'advice', 'need', 'assist', 'suggestion']):
        contextual_topics.append('Advice')
    
    # Media patterns
    if any(word in title_lower for word in ['pic', 'picture', 'photo', 'image', 'video', 'clip']):
        contextual_topics.append('Media')
        
    # Combine all topic sources
    all_potential_topics = list(set(contextual_topics + bigram_topics + tfidf_topics + common_words))
    
    # Format topics
    cleaned_topics = []
    for topic in all_potential_topics:
        # Convert to title case and remove non-alphanumerics
        clean_topic = ' '.join(word.capitalize() for word in re.sub(r'[^\w\s]', '', topic).split())
        if clean_topic and len(clean_topic) > 2:
            cleaned_topics.append(clean_topic)
    
    # Sort by likelihood of being a topic (contextual > bigrams > tfidf > common words)
    ranked_topics = []
    for topic in contextual_topics:
        if topic in cleaned_topics:
            ranked_topics.append(topic)
            cleaned_topics.remove(topic)
            
    for topic in bigram_topics:
        clean_topic = ' '.join(word.capitalize() for word in re.sub(r'[^\w\s]', '', topic).split())
        if clean_topic in cleaned_topics:
            ranked_topics.append(clean_topic)
            cleaned_topics.remove(clean_topic)
    
    # Add remaining topics
    ranked_topics.extend(cleaned_topics)
    
    # Select top topics
    final_topics = ranked_topics[:num_topics]
    
    # If no topics found, use generic defaults
    if not final_topics:
        final_topics = ['Discussion', 'General', 'Question', 'Opinion', 'Information']
    
    # Assign topics to each comment based on content similarity
    df['topic'] = assign_topics_to_comments(df, final_topics)
    
    return df, final_topics

def assign_topics_to_comments(df, topics, fallback=True):
    """
    Assign topics to comments based on content similarity
    
    Args:
        df: DataFrame with comments
        topics: List of extracted topics
        fallback: Whether to use random assignment as fallback
        
    Returns:
        Series with topic assignments
    """
    # Initialize list to hold assigned topics
    assigned_topics = []
    
    # For each comment, find most relevant topic
    for _, row in df.iterrows():
        comment_text = clean_text(row['text'])
        
        # Skip assignment for very short or empty comments
        if len(comment_text.split()) < 3:
            assigned_topics.append(np.random.choice(topics))
            continue
        
        # Score each topic by word overlap
        topic_scores = {}
        for topic in topics:
            # Convert topic to lowercase for comparison
            topic_words = set(topic.lower().split())
            # Calculate overlap
            overlap = sum(1 for word in comment_text.split() if word in topic_words)
            topic_scores[topic] = overlap
        
        # Find topic with highest score
        max_score = max(topic_scores.values())
        best_topics = [t for t, score in topic_scores.items() if score == max_score]
        
        # If no clear match, use random assignment
        if max_score == 0 or (len(best_topics) == len(topics) and fallback):
            assigned_topics.append(np.random.choice(topics))
        else:
            assigned_topics.append(np.random.choice(best_topics))
    
    return assigned_topics

def load_data(file_path=None, topic_extraction=True):
    """Enhanced load_data function with dynamic topic extraction"""
    # Original data loading logic
    if file_path is None:
        # Use sample data if no file is provided
        sample_json = """{'id': '1j8qvlw', 'title': 'From 2013-2019- Rohit scored 8 - 150s in ODI cricket. After 2019- 0 150s', 'author': 'Anxious-Progress3480', 'subreddit': 'IndiaCricket', 'score': 203, 'num_comments': 34, 'created_utc': 1741700102.0, 'selftext': '', 'topic': ['Video'], 'comments': ['#Join our official [Discord server](https://discord.gg/TXyybaSwRH) for more discussions.\\n\\n\\n*I am a bot, and this action was performed automatically. Please [contact the moderators of this subreddit](/message/compose/?to=/r/IndiaCricket) if you have any questions or concerns.*', 'He missed lot of 100s in 2023', 'In my dreams I still have a 150 partnership between RoKo on 19/11\\n\\nRohit was looking in such good flow and Kohli hammered Starc with 3 fours in a row', 'You did so much 😭\\n\\nhttps://preview.redd.it/viohvgjjp2oe1.png?width=1080&format=png&auto=webp&s=1b93376d3a3d9ee17d3478be4ecbee66b413d3e9', 'i used to expect a 200 every time he crosses his 100 at that period....', "Heads off to him I have seen all of his daddy hundreds. he can hit daddy hundreds still but choose to change the approach of team agressive cricket. Performing in 2 wc 2019 hitting 5 centuries and 2023 with completely different approaches unbelievable. Heads off to this guy he bring that all depth thing in 2023 world cup where he bring Shardul or Ashwin at 8 number but sadly Pandya got injured so we weren't able to continue that. But he continues that in 24 wc and 25 CT and we succeed. Rohit is genius captain In White bowl winning 2 wc dominating way is fabulous.", 'Ximbaba can never', "TBH, I feel like his selfless approach ( of hitting sixes in pp overs and capitalising the max. there) can only  work on Asian pitches.\\n\\nOn Pitches where the ball swings & moves, Sefless cricket is a bad idea, major rzn why He failed in BGT . his prev approach of 2019 WC is better, in such conditions.\\n\\nSince He is not retiring, He hope he realise dis too, We've Eng series coming up.", 'Meanwhile Kohli will take 20 balls in his 90s to complete his century.', 'MS Dhoni or Rohit Sharma? The debate continues! Dive into an in-depth comparison of their captaincy records, achievements, and leadership styles. 🔥\\n\\n👉 Read the full analysis here: [https://dailycurrent4you.blogspot.com/2025/03/rohit-sharma-vs-ms-dhoni-who-is.html', 'So is he saying that his batting approach was one of the reasons we lost 2019 wc?', 'Yeah 2-3 easily.', 'Kohli too missed 3 centuries in 2023 wc', '_Hats off_ buddy', 'Who?', 'he is not playing with a selfless approach in test matches he is just bad at countering seam.', 'Even after the performance he displayed in CT you are not satisfied? Get real dawg.', 'Man, your hate for Rohit is so ridiculously juvenile! Have you thought of doing something productive with your life?', 'Yeah rest of them scored double century in every match', "Except the one against Pakistan, I don't remember if he got any 80+ score", 'Who asked about kohli?', 'Ni, Kanpur test me ye selfless hi khel raha tha, Then it worked due to asian conditions, and yeaa He struggles with seam i agree!! \\n\\nbut he did pretty well wid his past approach in 2019 wc nd Eng series, even scored a 100 at one test.', "Wow if this is hate then let it be. Not my fault if Rohit fans aren't mature enough to comprehend a simple question.", 'So selfish of rest of them for scoring double centuries when our selfless Rohit was just scoring centuries', 'Against England...', 'Who told you?', "They were trying to chase the total after missing 2 days due to rain, and the opposition was Bangladesh. In the first innings, Kohli played at 134 sr, KL RAHUL PLAYED at 159sr. KL Rahul!! it's just a one match phenomenon, based on situation and opposition.", 'Dude! I just checked your comment history. Every other comment is about dissing Rohit and elevating Kohli.\\n\\nYou have an obsession. Get professional help. This kind of fixation with celebrities is not healthy.', 'The post*', 'they can keep crying kohli legacy as a captain is 0/nil  rohit changed the culture these kohli  schoolboy fans are crying since the day ganguly whoop his ass out and they just wanna hate him', 'Since you have so much time to check other\\'s comment history, maybe dig a little deeper to BGT days and check my comments "elevating" Kohli. \\n\\nAnd yeah obsession with celebrities is definitely not healthy, that\\'s the reason I\\'m not one of those retards who joins Bollywood gossip subs.', 'Then why asking me? Ask from the post*', "Whatever. I'm not the one taking piss at a world cup winning captain. You are. Now go see a therapist before it's too late."]}"""
        data = json.loads(sample_json.replace("""'""", '"'))
    else:
        # Load data from the provided file
        with open(file_path, 'r') as file:
            data = json.loads(file.read())
    
    # Process post data
    post_data = {
        'id': data['id'],
        'title': data['title'],
        'author': str(data['author']),
        'subreddit': str(data['subreddit']),
        'score': data['score'],
        'num_comments': data['num_comments'],
        'created_utc': datetime.fromtimestamp(data['created_utc']),
        'selftext': data['selftext'],
        'topics': data.get('topic', [])
    }
    
    # Process comments
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as Bert
    from model import make_pred
    from model import YayOrNay
    analyzer = Bert()
    comments_data = []
    
    for i, comment in enumerate(data['comments']):
        # Generate a random author name for demonstration
        author = f"user_{i}" if i > 0 else "AutoModerator"
        
        # Perform sentiment analysis on the comment
        # sentiment_score = analyzer.polarity_scores(comment)['compound']
        sentiment_score = make_pred(comment)
        
        # if sentiment_score >= 0.05:
        #     sentiment = 'Positive'
        # elif sentiment_score <= -0.05:
        #     sentiment = 'Negative'
        # else:
        #     sentiment = 'Neutral'
        # sentiment_score = make_pred(comment)
        
        # Determine sentiment category based on compound score
        if sentiment_score == 0:
            sentiment = 'Negative'   # sadness
        elif sentiment_score == 1:
            sentiment = 'Positive'   # joy
        elif sentiment_score == 2:
            sentiment = 'Positive'   # love
        elif sentiment_score == 3:
            sentiment = 'Negative'   # anger
        elif sentiment_score == 4:
            sentiment = 'Negative'   # fear
        else:
            sentiment = 'Neutral'  
        
        # Extract keywords (simple implementation - can be improved)
        words = re.findall(r'\b\w+\b', comment.lower())
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with']
        keywords = [word for word in words if len(word) > 3 and word not in common_words][:5]
        
        # Generate random engagement metrics for demonstration
        upvotes = np.random.randint(1, 100) if i > 0 else post_data['score']
        
        comments_data.append({
            'comment_id': f"comment_{i}",
            'post_id': post_data['id'],
            'author': author,
            'text': comment,
            'date': post_data['created_utc'] + timedelta(minutes=np.random.randint(1, 60*24)),  # Random time after post
            'upvotes': upvotes,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'keywords': keywords,
            # Generate random user attributes for demonstration
            'user_karma': np.random.randint(100, 10000),
            'user_age_days': np.random.randint(30, 3650),
        })
    
    # Create DataFrame from comments
    df = pd.DataFrame(comments_data)
    
    # Add post information to DataFrame for reference
    df['post_title'] = post_data['title']
    df['post_author'] = post_data['author']
    df['post_score'] = post_data['score']
    df['subreddit'] = post_data['subreddit']
    
    # Dynamic topic extraction if enabled
    if topic_extraction:
        df, extracted_topics = extract_topics(df, post_data)
        post_data['extracted_topics'] = extracted_topics
    else:
        # Fallback to random topics
        topics = ['Discussion', 'General', 'Question', 'Opinion', 'Information']
        df['topic'] = np.random.choice(topics, len(df))
    
    return df, post_data

# Example usage with any JSON data
def process_json_data(file_path):
    """Process arbitrary JSON data and extract topics"""
    data = {}
    with open(file_path, 'r', errors="ignore") as file:
        data = json.loads(file.read())
    
    
    # Process post data
    post_data = {
        'id': data['id'],
        'title': data['title'],
        'author': str(data['author']),
        'subreddit': str(data['subreddit']),
        'score': data['score'],
        'num_comments': data['num_comments'],
        'created_utc': datetime.fromtimestamp(data['created_utc']),
        'selftext': data['selftext'],
        'topics': data.get('topic', [])
    }
    
    # Process comments with sentiment analysis
    from nltk.sentiment.vader import SentimentIntensityAnalyzer as Bert
    from model import make_pred
    from model import YayOrNay
    model = Bert()
    comments_data = []
    
    for i, comment in enumerate(data['comments']):
        author = f"user_{i}" if i > 0 else "AutoModerator"
        # sentiment_score = model.polarity_scores(comment)['compound']
        sentiment_score = make_pred(comment)

        # if sentiment_score >= 0.05:
        #     sentiment = 'Positive'
        # elif sentiment_score <= -0.05:
        #     sentiment = 'Negative'
        # else:
        #     sentiment = 'Neutral'

        if sentiment_score == 0:
            sentiment = 'Negative'   # sadness
        elif sentiment_score == 1:
            sentiment = 'Positive'   # joy
        elif sentiment_score == 2:
            sentiment = 'Positive'   # love
        elif sentiment_score == 3:
            sentiment = 'Negative'   # anger
        elif sentiment_score == 4:
            sentiment = 'Negative'   # fear
        else:
            sentiment = 'Neutral'    # surprise


        
        words = re.findall(r'\b\w+\b', comment.lower())
        common_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with']
        keywords = [word for word in words if len(word) > 3 and word not in common_words][:5]
        
        upvotes = np.random.randint(1, 100) if i > 0 else post_data['score']
        
        comments_data.append({
            'comment_id': f"comment_{i}",
            'post_id': post_data['id'],
            'author': author,
            'text': comment,
            'date': post_data['created_utc'] + timedelta(minutes=np.random.randint(1, 60*24)),
            'upvotes': upvotes,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment,
            'keywords': keywords,
            'user_karma': np.random.randint(100, 10000),
            'user_age_days': np.random.randint(30, 3650),
        })
    
    # Create DataFrame from comments
    df = pd.DataFrame(comments_data)
    
    # Add post information to DataFrame
    df['post_title'] = post_data['title']
    df['post_author'] = post_data['author']
    df['post_score'] = post_data['score']
    df['subreddit'] = post_data['subreddit']
    
    # Use our extract_topics function
    df, extracted_topics = extract_topics(df, post_data)
    post_data['extracted_topics'] = extracted_topics
    
    return df, post_data