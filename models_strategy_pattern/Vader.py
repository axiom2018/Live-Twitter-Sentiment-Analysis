import streamlit as st
from models_strategy_pattern.model_types import ModelTypes
from models_strategy_pattern.model_strategy import ModelStrategy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

''' 
            Vader

    This is a much simplier model to use in this project. The library is easy to get the hang of and will 
    produce good results.

'''

class Vader(ModelStrategy):
    def __init__(self) -> None:
        super().__init__(ModelTypes.SimpleLibrary)
        self.m_vader_model = None
    


    @st.cache(allow_output_mutation=True)
    def Load(self):
        self.m_vader_model = SentimentIntensityAnalyzer()


    
    def Predict(self, num_of_tweets_to_display=5, values_in_each_tweet=10):
        st.write('Predict!')