import re
import sqlite3
import pandas
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

database_file = "minisocial_database.sqlite"
try:
    conn = sqlite3.connect(database_file)
    print("All is OK")
except Exception as e:
    print("Something went wrong")


#EXERCISE 4.1 POPULAR TOPICS
def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

    content_from_all = """select id, content, 'post' as table_name from posts 
                        union all select id, content, 'comment' as table_name from comments"""
    content_data = pandas.read_sql_query(content_from_all, conn)
    stop_words = stopwords.words('english')
    stop_words.extend(['would', 'best', 'always', 'amazing', 'bought', 'quick', 'haha', 'like',
                       'people', 'new', 'fun', 'think', 'know', 'believe', 'many', 
                       'thing', 'need', 'small', 'even', 'make', 'love', 'mean', 
                       'fact', 'question', 'time', 'reason', 'also', 'could', 'true', 
                       'well',  'life', 'said', 'year', 'going', 'good', 'really', 
                       'much', 'want', 'back', 'look', 'article', 'host', 'university', 
                       'reply', 'thanks', 'mail', 'post', 'please'])
    lemmatizer = WordNetLemmatizer()
    bow_list = []
    for _, row in content_data.iterrows():
        text = row['content']
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text) #remove urls
        text = re.sub(r"[^a-z\s]", "", text) #remove punctuation and numbers
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 3]
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        if len(tokens) > 0:
            bow_list.append(tokens)

    dictionary = Dictionary(bow_list)
    dictionary.filter_extremes(no_below=3, no_above=0.3)
    corpus = [dictionary.doc2bow(tokens) for tokens in bow_list]
    optimal_coherence = -100
    optimal_lda = None
    optimal_k = 0
    for K in [100,125,150,175,200,225,250]:
        lda = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=10, random_state=2)
        coherence_model = CoherenceModel(model=lda, texts=bow_list, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        if(coherence_score > optimal_coherence):
            print(f'LDA was trained with {K} topics. {coherence_score} as an average topic coherence score is the best so far!')
            optimal_coherence = coherence_score
            optimal_lda = lda
            optimal_k = K
        else: 
            print(f'LDA was trained with {K} topics. {coherence_score} as an average topic coherence score is not that good.')

    analyzer = SentimentIntensityAnalyzer()
    topic_counts = [0] * optimal_k
    topic_sentiments = [[]for _ in range(optimal_k)]
    for i, bow in enumerate(corpus):
        topic_dist = optimal_lda.get_document_topics(bow)
        if not topic_dist:
            continue
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]
        if dominant_topic < len(topic_counts):
            topic_counts[dominant_topic] += 1
            text = " ".join(bow_list[i])
            sentiment = analyzer.polarity_scores(text)["compound"]
            topic_sentiments[dominant_topic].append(sentiment)
    sorted_topics = sorted(enumerate(topic_counts), key=lambda x: x[1], reverse=True)
    print('Top 10 topics by frequency together with 10 words representing the topic: ')
    for i, (topic_index, count) in enumerate(sorted_topics[:10], start=1):
        topic_words = optimal_lda.show_topic(topic_index, topn=10)
        topic_keywords = ", ".join([w for w, _ in topic_words])
        if topic_sentiments[topic_index]:
            average_sentiment = sum(topic_sentiments[topic_index]) / len(topic_sentiments[topic_index])
            value = "POSITIVE" if average_sentiment > 0.05 else "NEGATIVE" if average_sentiment < -0.05 else "NEUTRAL"
            sentiment_information = f"{average_sentiment:.2f} ({value})"
        print(f'{i}. Topic {topic_index} ({count} posts): {topic_keywords}')
        print(f'Average sentiment score: {sentiment_information}')

if __name__ == '__main__':
    main()