import streamlit as st
from models_strategy_pattern.model_types import ModelTypes


''' 
        Model Strategy

This class is a base class to all models that will derive from it. It may need to be updated for different/new types
of models that will be later added.


'''

class ModelStrategy:
    def __init__(self, type_of_model, model_name) -> None:
        # Checking what the type of model is for each class in the list in main.py is vital. It'll check if a particular model belongs in that class.
        self.m_type = type_of_model

        self.m_model_name = model_name

        # A quick explanation on a model will help the user gain understanding.
        self.m_model_details = ''' '''

    

    ''' 

            All functions below are for derived classes to override.


            1) Load - Of course loading the model.


            2) Predict - 

                a) example_tweets_list - This is the main list that's provided for 2 reasons. One is to get the example plot up and displaying of course.
                    The other is to help the tweets from twitter have a real "foothold" regarding other tweets that are positive, neutral, or negative.
                    Assuming that no simplier library like Vader is selected at the beginning of the project.

                b) new_tweets_from_twitter_list - All the tweets pulled from Twitter in the "get_tweets.py" file.

                c) num_of_tweets_to_display - There will be a lot of embeddings, how many to display to the user.

                d) values_in_each_tweets - The embeddings are ridiculously long. How many values to show to the user?
    
    '''

    def Load(self):
        pass


    def Predict(self, example_tweets_list, new_tweets_from_twitter_list=None, num_of_tweets_to_display=5, values_in_each_tweet=10):
        pass





    ''' 

                All functions below are implemented in this class. All dervived objects need not override.
    
    '''

    # Used in data.py to check proper type of a model, seeing it the current page is one it belongs on.
    def GetType(self):
        return self.m_type


    # When selecting a model in model_selection.py, this function is used to display the name.
    def GetName(self):
        return self.m_model_name