
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np
from keybert import KeyBERT

class KeywordExtractor:
    def __init__(self, method='tfidf'):
        self.method = method
        if method == 'spacy':
            self.nlp = spacy.load('en_core_web_sm')
        elif method == 'keybert':
            self.model = KeyBERT()
    
    def extract_keywords_tfidf(self, text, n_keywords):
        """TF-IDF based keyword extraction"""
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get sorted indices of TF-IDF scores
        dense = tfidf_matrix.todense()
        scores = np.asarray(dense)[0]
        sorted_indices = np.argsort(scores)[::-1]
        
        return [feature_names[idx] for idx in sorted_indices[:n_keywords]]
    
    def extract_keywords_spacy(self, text, n_keywords):
        """Spacy based keyword extraction"""
        doc = self.nlp(text)
        # Extract noun phrases and named entities
        keywords = []
        keywords.extend([chunk.text for chunk in doc.noun_chunks])
        keywords.extend([ent.text for ent in doc.ents])
        
        # Count frequencies and get top keywords
        keyword_freq = Counter(keywords)
        return [kw for kw, _ in keyword_freq.most_common(n_keywords)]
    
    def extract_keywords_keybert(self, text, n_keywords):
        """KeyBERT based keyword extraction"""
        keywords = self.model.extract_keywords(text, 
                                            keyphrase_ngram_range=(1, 2),
                                            stop_words='english',
                                            top_n=n_keywords)
        return [kw for kw, _ in keywords]
    
    def extract(self, text, n_keywords):
        if self.method == 'tfidf':
            return self.extract_keywords_tfidf(text, n_keywords)
        elif self.method == 'spacy':
            return self.extract_keywords_spacy(text, n_keywords)
        elif self.method == 'keybert':
            return self.extract_keywords_keybert(text, n_keywords)
