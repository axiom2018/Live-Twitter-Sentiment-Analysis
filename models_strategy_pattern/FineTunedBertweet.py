import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np

from models_strategy_pattern.model_strategy import ModelStrategy
from models_strategy_pattern.model_types import ModelTypes
from clustering_strategy_pattern.k_means import SKL_KMeans
from Etc.Embeddings import Embeddings
from Etc.ScatterPlot import ScatterPlot
from Etc.Spacing import Spacing
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


''' 
            Fine tuned bertweet sentiment analysis

    This model is based on a different model which was trained on 850 million tweets, some covid related. It's chosen
    because it works within the labels of this project, negative, neutral, and positive (in that order). 
    
'''

class FineTunedBertweet(ModelStrategy, Spacing):
    def __init__(self):
        super().__init__(ModelTypes.SimpleLibrary, "Bert Base Uncased")

        # This model will need embedding management, so let the class handle that.
        self.m_embeddings_manager = Embeddings()

        # Clustering is necessary for embeddings turned into 2d values.
        self.m_clustering_algorithm = SKL_KMeans(1234)

        # Displaying the visual graph.
        self.m_plot_manager = ScatterPlot()

        # Set to null by default.
        self.m_model = None
        self.m_tk = None

        self.m_model_details = ''' This is a huggingface model that is based on a bigger model that was trained on over 800 million english tweets, some
        even covid related. It fits in well with the default set of labels in the project which are positive, neutral, and negative. However it's own
        **order** of those labels is negative, neutral, and positive. It's simple to use and very effective. '''


    '''


        Functions for Loading.

    
    '''

    # Multiple things need to be loaded for the model to be ready, this function handles it all. 
    @st.cache(allow_output_mutation=True)
    def Load(self):
        self.LoadModel()
        self.LoadTokenizer()


    def LoadTokenizer(self):
        self.m_tk = AutoTokenizer.from_pretrained("rabindralamsal/finetuned-bertweet-sentiment-analysis")
        # self.m_tk = BertTokenizer.from_pretrained('bert-base-uncased')


    def LoadModel(self):
        self.m_model = TFAutoModelForSequenceClassification.from_pretrained("rabindralamsal/finetuned-bertweet-sentiment-analysis")
        # self.m_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)


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





        self.ApplySpacingOnScreen(5)
        st.write(f'Testing {self.GetName()} model')

        for i in range(num_of_tweets_to_display):
            tweet = df['Cleaned Tweets'][i]

            # Display tweet.
            st.write(f'Tweet {i+1}) {tweet}')

            # Encode the tweet, turning it into numbers.
            encoded_tweet = self.m_tk.encode(tweet, return_tensors='tf')
            st.write(f'Encoded tweet: {encoded_tweet}')

            #
            raw_model_prediction = self.m_model.predict(encoded_tweet)[0]
            st.write(f'Raw model prediction: {raw_model_prediction}')

            # Get the floating point values that represent each of the labels.
            prediction = tf.nn.softmax(raw_model_prediction, axis=1).numpy()
            st.write(f'Prediction: {prediction}')

            # Now the real sentiment in the order negative, neutral, positive, in float form.
            sentiment = np.argmax(prediction)
            st.write(f'---Sentiment---')
            st.write(f'{sentiment}')


            self.ApplySpacingOnScreen(3)


