import pickle
import os
import json

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

def process(dataset):
    with open(f'utils/tmp/{dataset}_graph.pkl', 'rb') as f:
        graphs = pickle.load(f)
    print(len(graphs))
    print(graphs[0])
    
    with open(f'data/{dataset}/raw/{dataset[:3]}_id_twitter_mapping.pkl', 'rb') as f:
        twitter_mapping = pickle.load(f)
    print(twitter_mapping[3])

if __name__ == "__main__":
    dataset = 'politifact'
    # dataset = 'gossipcop'
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