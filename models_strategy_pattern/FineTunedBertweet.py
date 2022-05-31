import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

from models_strategy_pattern.model_strategy import ModelStrategy
from models_strategy_pattern.model_types import ModelTypes
from Etc.Spacing import Spacing
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


''' 
            Fine tuned bertweet sentiment analysis

    This model is based on a different model which was trained on 850 million tweets, some covid related. It's chosen
    because it works within the labels of this project, negative, neutral, and positive (in that order). 
    
'''

class FineTunedBertweet(ModelStrategy, Spacing):
    def __init__(self):
        super().__init__(ModelTypes.SimpleLibrary, "Fine Tuned Bertweet")

        self.m_model_details = ''' This is a huggingface model that is based on a bigger model that was trained on over 800 million 
        english tweets, some even covid related. It fits in well with the default set of labels in the project which are positive, 
        neutral, and negative. However it's own **order** of those labels is negative, neutral, and positive. The model will use 
        integers in place of the aforementioned **string** versions like 0/negative, 1/neutral, 2/positive. '''

        # Set to null by default, will be set when Load function is called.
        self.m_model = None
        self.m_tokenizer = None

        # Used to convert int prediction to string.
        self.m_list_of_labels = ['Negative', 'Neutral', 'Positive']

        # Dataframes will be created in the CreateDataframesOfTweets function. Each will be saved with these variables.
        self.m_df_neg_tweets = None
        self.m_df_neu_tweets = None
        self.m_df_pos_tweets = None

        # Message to be displayed right before dataframes are shown.
        self.m_results_message = f''' Below are dataframes of the of the tweets that the Fine Tuned Bertweet model has identified as 
        negative, neutral, or positive. It returns 3 floating point numbers which represent how much the model thinks the given tweets
        represent each of the aforementioned string labels. Getting the highest of the 3 and converting to a string. '''


    '''


        Functions for Loading.

    
    '''

    # Multiple things need to be loaded for the model to be ready, this function handles it all. 
    @st.cache(allow_output_mutation=True)
    def Load(self):
        self.LoadModel()
        self.LoadTokenizer()



    def LoadTokenizer(self):
        self.m_tokenizer = AutoTokenizer.from_pretrained("rabindralamsal/BERTsent")



    def LoadModel(self):
        self.m_model = TFAutoModelForSequenceClassification.from_pretrained("rabindralamsal/BERTsent")



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

        # Display a couple of predictions with more detail to it.
        self.ApplySpacingOnScreen(5)
        st.write(f'### {num_of_tweets_to_display} tweet predictions (in detail).')

        for i in range(num_of_tweets_to_display):
            tweet = df['Cleaned Tweets'][i]

            # Display tweet.
            st.write(f'##### Tweet {i+1}')
            st.write(tweet)

            # Encode the tweet, turning it into numbers.
            encoded_tweet = self.m_tokenizer.encode(tweet, return_tensors='tf')
            st.write(f'Encoded tweet using tokenizer: {encoded_tweet}')

            #
            raw_model_prediction = self.m_model.predict(encoded_tweet)[0]

            # Get the floating point values that represent each of the labels.
            prediction = tf.nn.softmax(raw_model_prediction, axis=1).numpy()
            st.write(f'Prediction: {prediction}')

            # Now the real sentiment in the order negative, neutral, positive, in float form.
            sentiment_number = np.argmax(prediction)
            st.write(f'{self.GetName()} predicted {self.GetStringLabel(sentiment_number)}')

            self.ApplySpacingOnScreen(3)

        # Create the dataframes of each set of predicted tweets. Negative, neutral, and positive.
        self.ApplySpacingOnScreen(2)
        self.CreateDataframesOfTweets(df)
        self.DisplayDataframes()



    # Convert the number that model prediction gives into a string so the results make more sense for the user.
    def GetStringLabel(self, number):
        string_to_return = None

        if number == 0:
            string_to_return = 'Negative'
        elif number == 1:
            string_to_return = 'Neutral'
        elif number == 2:
            string_to_return = 'Positive'
        
        return string_to_return


    
    def CreateDataframesOfTweets(self, dataframe):
        # For the dataframes to soon be created.
        column_names = ['Tweets', 'Label']

        # Create lists for each separate dataframe.
        pos_tweet_list = []
        neu_tweet_list = []
        neg_tweet_list = []

        length = len(dataframe.index)

        for i in range(length):
            tweet = dataframe['Cleaned Tweets'][i]

            # Encode the tweet, turning it into numbers.
            encoded_tweet = self.m_tokenizer.encode(tweet, return_tensors='tf')

            #
            raw_model_prediction = self.m_model.predict(encoded_tweet)[0]

            # Get the floating point values that represent each of the labels.
            prediction = tf.nn.softmax(raw_model_prediction, axis=1).numpy()

            # Get the sentiment string.
            sentiment_string = self.GetStringLabel(np.argmax(prediction))

            # Get a new dataframe row.
            new_df_row = {column_names[0]: tweet,
                          column_names[1]: sentiment_string}
            
            # Check what kind of sentiment string it is, and add it to the appropriate list.
            if sentiment_string == self.m_list_of_labels[0]:
                neg_tweet_list.append(new_df_row) # Negative.
            elif sentiment_string == self.m_list_of_labels[1]:
                neu_tweet_list.append(new_df_row) # Neutral.
            elif sentiment_string == self.m_list_of_labels[2]:
                pos_tweet_list.append(new_df_row) # Positive.
        

        # Create dataframes for each type of tweet (neg, neu, pos) to be held.
        df_neg_tweets = pd.DataFrame(neg_tweet_list, columns=column_names)
        df_neu_tweets = pd.DataFrame(neu_tweet_list, columns=column_names)
        df_pos_tweets = pd.DataFrame(pos_tweet_list, columns=column_names)

        # Save the dataframes.
        self.m_df_neg_tweets = df_neg_tweets
        self.m_df_neu_tweets = df_neu_tweets
        self.m_df_pos_tweets = df_pos_tweets

    

    def DisplayDataframes(self):
        self.ApplySpacingOnScreen(5)
        st.write('### Description of the results: ')
        st.write(self.m_results_message)
        self.ApplySpacingOnScreen(3)

        st.write('#### Dataframe of positive tweets: ')
        st.write(self.m_df_pos_tweets)
        self.ApplySpacingOnScreen(5)

        st.write('#### Dataframe of neutral tweets: ')
        st.write(self.m_df_neu_tweets)
        self.ApplySpacingOnScreen(5)

        st.write('#### Dataframe of negative tweets: ')
        st.write(self.m_df_neg_tweets)
        self.ApplySpacingOnScreen(5)