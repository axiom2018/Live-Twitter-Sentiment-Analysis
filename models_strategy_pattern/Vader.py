import streamlit as st
import pandas as pd
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
        super().__init__(ModelTypes.SimpleLibrary, "Vader")
        self.m_vader_model = None

        self.m_model_details = ''' Vader is a rule based sentiment analysis model that is **_specifically_** tuned for social media.
        It give scores regarding 3 kinds of sentiment for each given string. Those are positive, neutral, and negative. Very convenient 
        that those are the sentiment labels this project of course uses, which is why it's used here. Vader is incredibly simple to
        use in code as well. '''
    


    @st.cache(allow_output_mutation=True)
    def Load(self):
        self.m_vader_model = SentimentIntensityAnalyzer()


    
    def Predict(self, num_of_tweets_to_display=5, values_in_each_tweet=10):
        st.title(f'{self.m_model_name} predictions.')
        self.ApplySpacingOnScreen(3)

        st.write(self.m_model_details)
        self.ApplySpacingOnScreen(3)

        # Show some information about the tweets to be displayed.
        total_tweets = st.session_state.dataframe_of_tweets['Cleaned Tweets'].tolist()
        len_total_tweets = len(total_tweets)
        st.write(f'### {num_of_tweets_to_display} tweets shown out of {len_total_tweets}.')

        # Get the actual dataframe of tweets that contain the new tweets pulled from twitter.
        df = st.session_state.dataframe_of_tweets

        # Show a few of the tweets to the user. The ones that will be used for prediction.
        for i in range(num_of_tweets_to_display):
            tweet = df['Cleaned Tweets'][i]
            st.write(f'{i+1}) {tweet}')
            st.write('')

        # For the dataframes to soon be created.
        column_names = ['Tweets', 'Label', 'Vader polarity score (in "neg, neu, pos" order)']

        # Create lists for the dataframes to be later created.
        pos_tweet_list = []
        neu_tweet_list = []
        neg_tweet_list = []

        # The sentiment in the dictionaries returned by polarity_scores will be in the same order.
        # So this list of full name (for example, pos is positive) labels will be used.
        labels = ['negative', 'neutral', 'positive']
        
        # Now begin predictions
        for i in range(10):
            # Get the scores of the current tweet at index i. This will return a dictionary.
            sentiment_of_tweet = self.m_vader_model.polarity_scores(total_tweets[i])

            # With that dictionary, it will always be in the form "'neg', 'neu', 'pos'". That's
            # good because it's possible to extract all the sentiment floating point scores and
            # be able to know which goes with which type of sentiment. Start by getting the values
            # in a list.
            values_list = list(sentiment_of_tweet.values())

            # The last element of the list is a "compound" value that polarity_scores gives. It's not
            # needed, so remove it.
            values_list = values_list[:-1]

            # Get the index of the highest value. Since the labels list created earlier and the labels
            # inside the returned polarity_scores dictionary MATCH, it means that it's possible to 
            # get the right full name label with the index of the highest value.
            highest_value = max(values_list)
            highest_value_index = values_list.index(highest_value)
            label = labels[highest_value_index]

            # Get the dictionary ready, this will be data made of the current tweet in the loop and will
            # be added to a row in the appropriate dataframe.
            new_df_row = {column_names[0]: total_tweets[i],
                          column_names[1]: label,
                          column_names[2]: values_list}

            # Check which label it is. For all types, create a pro
            if label == labels[0]:
                neg_tweet_list.append(new_df_row) # Negative.
            elif label == labels[1]:
                neu_tweet_list.append(new_df_row) # Neutral.
            elif label == labels[2]:
                pos_tweet_list.append(new_df_row) # Positive
            
            
            # self.ApplySpacingOnScreen(3)

        
        # Create dataframes for each type of tweet (pos, neu, neg) to be held.
        df_pos_tweets = pd.DataFrame(pos_tweet_list, columns=column_names)
        df_neu_tweets = pd.DataFrame(neu_tweet_list, columns=column_names)
        df_neg_tweets = pd.DataFrame(neg_tweet_list, columns=column_names)

        st.write('### Dataframe of positive tweets: ')
        st.write(df_pos_tweets)
        self.ApplySpacingOnScreen(5)

        st.write('### Dataframe of neutral tweets: ')
        st.write(df_neu_tweets)
        self.ApplySpacingOnScreen(5)

        st.write('### Dataframe of negative tweets: ')
        st.write(df_neg_tweets)
        self.ApplySpacingOnScreen(5)