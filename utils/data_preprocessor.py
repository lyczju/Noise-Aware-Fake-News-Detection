import pickle
import os
import json
import re
import string
import pandas as pd
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
from emoji import demojize
from keyword_extractor import KeywordExtractor
from similarity_matcher import SimilarityMatcher

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def extract_tweets(dataset):
    with open(f'data/{dataset}/raw/{dataset[:3]}_id_time_mapping.pkl', 'rb') as f:
        time_mapping = pickle.load(f)

    with open(f'data/{dataset}/raw/{dataset[:3]}_id_twitter_mapping.pkl', 'rb') as f:
        twitter_mapping = pickle.load(f)

    graphs = dict()
    n = len(twitter_mapping)
    idx = 0
    while idx < n:
        if dataset in twitter_mapping[idx]:
            root_idx = idx
            user_id_list = []
            idx += 1
            user_id = twitter_mapping[idx]
            while dataset not in user_id:
                user_id_list.append(user_id)
                idx += 1
                if idx >= n:
                    break
                user_id = twitter_mapping[idx]
            graphs[root_idx] = user_id_list


    # Find the correct directory for each key
    for key in graphs.keys():
        event = twitter_mapping[key]
        fake_dir = os.path.join('data', 'FakeNewsNet', f'{dataset}_fake', event)
        real_dir = os.path.join('data', 'FakeNewsNet', f'{dataset}_real', event)
        
        target_dir = None
        label = None
        if os.path.exists(fake_dir):
            target_dir = fake_dir
            label = 'fake'
        elif os.path.exists(real_dir):
            target_dir = real_dir
            label = 'real'
            
        if target_dir:
            # Read tweets.json
            tweets_file = os.path.join(target_dir, 'tweets.json')
            if os.path.exists(tweets_file):
                with open(tweets_file, 'r') as f:
                    tweet_data = json.load(f)
                    
                # Create dict to store tweets by user
                event_users = {}
                
                # Get user_ids from graphs value
                user_ids = graphs[key]
                
                # Match tweets with user_ids
                for user_id in user_ids:
                    user_tweets = []
                    for tweet in tweet_data['tweets']:
                        if str(tweet['user_id']) == str(user_id):
                            user_tweets.append(tweet)
                    if len(user_tweets) > 0:
                        event_users[user_id] = user_tweets
                            
                # Update graphs value
                graphs[key] = event_users

    with open(f'{dataset}_graph.pkl', 'wb') as f:
        pickle.dump(graphs, f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = demojize(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def process(dataset):
    # 1. Clean texts
    # with open(f'utils/tmp/{dataset}_graph.pkl', 'rb') as f:
    #     graphs = pickle.load(f)
    # for root_id, user in graphs.items():
    #     for user_id, tweets in user.items():
    #         cleaned_tweets = []
    #         for tweet in tweets:
    #             cleaned_tweets.append(clean_text(tweet['text']))
    #         user[user_id] = cleaned_tweets
    
    # with open(f'utils/tmp/{dataset}_graph_cleaned.pkl', 'wb') as f:
    #     pickle.dump(graphs, f)
    
    with open(f'utils/tmp/{dataset}_graph_cleaned.pkl', 'rb') as f:
        graphs = pickle.load(f)
        
    with open(f'data/{dataset}/raw/{dataset[:3]}_id_twitter_mapping.pkl', 'rb') as f:
        twitter_mapping = pickle.load(f)
        
    # 2. Extract articles and summarize keywords
    # news_articles = {}
    # for key in graphs.keys():
    #     event = twitter_mapping[key]
    #     fake_dir = os.path.join('../data', 'FakeNewsNet', f'{dataset}_fake', event)
    #     real_dir = os.path.join('../data', 'FakeNewsNet', f'{dataset}_real', event)
        
    #     target_dir = None
    #     if os.path.exists(fake_dir):
    #         target_dir = fake_dir
    #     elif os.path.exists(real_dir):
    #         target_dir = real_dir
            
    #     if target_dir:
    #         # Read tweets.json
    #         tweets_file = os.path.join(target_dir, 'news_article.json')
    #         if os.path.exists(tweets_file):
    #             with open(tweets_file, 'r') as f:
    #                 article = json.load(f)
        
    #     if 'text' in article.keys():
    #         news_articles[key] = clean_text(article['text'])
    #     else:
    #         news_articles[key] = ''
            
    # with open(f'utils/tmp/{dataset}_news_articles.pkl', 'wb') as f:
    #     pickle.dump(news_articles, f)
    
    # with open(f'utils/tmp/{dataset}_news_articles.pkl', 'rb') as f:
    #     news_articles = pickle.load(f)
    
    # extractor = KeywordExtractor(method='keybert')
    # for k, v in news_articles.items():
    #     news_articles[k] = extractor.extract(v, n_keywords=4)
    
    # with open(f'utils/tmp/{dataset}_news_articles.pkl', 'wb') as f:
    #     pickle.dump(news_articles, f)
    
    with open(f'utils/tmp/{dataset}_news_articles.pkl', 'rb') as f:
        news_articles = pickle.load(f)
        
    # 3. Calculate similarities
    # reverse_twitter_mapping = defaultdict(list)
    # for k, v in twitter_mapping.items():
    #     reverse_twitter_mapping[v].append(k)

    # start_ids = []
    # end_ids = {}
    # for k in graphs.keys():
    #     start_ids.append(k)
    # for i in range(len(start_ids)):
    #     if i < len(start_ids) - 1:
    #         end_ids[start_ids[i]] = start_ids[i+1]    
    # end_ids[start_ids[-1]] = len(twitter_mapping)
    
    # matcher = SimilarityMatcher()
    # user_similarity_rankings = {}
    
    # for article_id in graphs.keys():
    #     keywords = news_articles[article_id]
    #     if not keywords:  # Skip if no keywords
    #         continue
            
    #     # Get all users and their tweets for this article
    #     user_tweets = {}  # Map user_id to all their tweets
    #     user_similarities = {}  # Map user_id to average similarity score
        
    #     for user_id, tweets in graphs[article_id].items():
    #         if not tweets:  # Skip users with no tweets
    #             continue
    #         user_tweets[user_id] = tweets
            
    #         # Calculate similarity between keywords and all tweets of this user
    #         similarity_matrix = matcher.calculate_similarity_matrix(keywords, tweets)
    #         # Take average of all similarity scores in the matrix
    #         avg_similarity = similarity_matrix.mean()
    #         user_similarities[user_id] = avg_similarity
        
    #     # Sort users by similarity score in descending order
    #     sorted_users = sorted(user_similarities.items(), 
    #                         key=lambda x: x[1], 
    #                         reverse=True)
        
    #     # Store only the sorted user IDs
    #     sorted_ids = []
    #     for user_id, sim in sorted_users:
    #         user_id_list = reverse_twitter_mapping[user_id]
    #         for u in user_id_list:
    #             if article_id < u < end_ids[article_id]:
    #                 sorted_ids.append((u, sim))
    #                 break
    #     user_similarity_rankings[article_id] = sorted_ids
    
    # with open(f'utils/tmp/{dataset}_user_similarity_rankings.pkl', 'wb') as f:
    #     pickle.dump(user_similarity_rankings, f)
        
    with open(f'utils/tmp/{dataset}_user_similarity_rankings.pkl', 'rb') as f:
        user_similarity_rankings = pickle.load(f)
    

if __name__ == "__main__":
    dataset = 'politifact'
    # dataset = 'gossipcop'
    process(dataset)
    # extract_tweets(dataset)
    
    
    # test
    # with open(f'{dataset}_graph.pkl', 'rb') as f:
    #     graphs = pickle.load(f)

    # cnt_users = 0
    # cnt_tweets = 0
    # for k, v in graphs.items():
    #     for k2, v2 in v.items():
    #         cnt_users += 1
    #         cnt_tweets += len(v2)