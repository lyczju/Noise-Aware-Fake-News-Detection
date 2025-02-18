from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def calculate_similarity_matrix(self, keywords, tweets):
        """Calculate similarity matrix between keywords and tweets"""
        # Encode keywords and tweets
        keyword_embeddings = self.model.encode(keywords)
        tweet_embeddings = self.model.encode(tweets)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(keyword_embeddings, tweet_embeddings)
        return similarity_matrix