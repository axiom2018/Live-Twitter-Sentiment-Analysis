import streamlit as st
from models_strategy_pattern.model_types import ModelTypes


''' 
        Model Strategy

This class is a base class to all models that will derive from it. It may need to be updated for different/new types
of models that will be later added.


'''

class ModelStrategy:
    def __init__(self, type_of_class, model_name) -> None:
        self.m_type = type_of_class
        self.m_model_name = model_name
        self.m_model_details = ''' '''

    

    ''' 

            All functions below are for derived classes to override.




            4) Predict - 

                a) example_tweets_list - This is the main list that's provided for 2 reasons. One is to get the example plot up and displaying of course.
                    The other is to help the tweets from twitter have a real "foothold" regarding other tweets that are positive, neutral, or negative.
                    Assuming that no simplier library like Vader is selected at the beginning of the project.

                b) new_tweets_from_twitter_list - All the tweets pulled from Twitter in the "get_tweets.py" file.

                c) num_of_tweets_to_display - There will be a lot of embeddings, how many to display to the user.

                d) values_in_each_tweets - The embeddings are ridiculously long. How many values to show to the user?
    
    '''

    def Load(self):
        pass


    def EmbedTweets(self):
        pass
    

    def DisplayTweetEmbeddings(self):
        pass


    def Predict(self, example_tweets_list, new_tweets_from_twitter_list=None, num_of_tweets_to_display=5, values_in_each_tweet=10):
        pass





    ''' 

                All functions below are implemented in this class. All dervived objects need not override.
    
    '''

    def GetType(self):
        return self.m_type


    # Referenced in model_selection.py
    def GetName(self):
        return self.m_model_name


    def GetModelType(self):
        return self.m_type


    # Help give spaces between portions of text and such on screen.
    def ApplySpacingOnScreen(self, amount_of_spaces):
        for i in range(amount_of_spaces):
            st.write('')