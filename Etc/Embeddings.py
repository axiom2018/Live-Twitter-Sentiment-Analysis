from turtle import distance
import streamlit as st
import umap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from sklearn.cluster import KMeans
from Etc.ScatterPlot import ScatterPlot



'''
            Embeddings.py

This class will mainly be used for models such as St_AllMpnetBaseV2.py and of course any other model that needs it. The current
St_AllMpnetBaseV2.py code has a lot of functions regarding embeddings as of 4/12/2022 and all this class will do is take that 
responsibility from it. Making the code in St_AllMpnetBaseV2.py smaller, cleaner, and also exercising single responsibility
principle. 


            ---Important!---

The models get declared in separate classes. For example in St_AllMpnetBaseV2.py, it's instantiated and saved in a session state.
Session states make it easier to handle variables in arbitrary parts of a streamlit project. If the name of the variable in the
line of code "if '(name here)' not in st.session_state:" changes in a models "Load" function, then this class won't run. That's the
only key issue to keep in mind because this class will reference whatever is in the "(name here)" part.


'''


class Embeddings:
    ''' Constructor variables:

        m_model - Save the model in the session_state everytime DisplayTweets is called.

        m_embedded_tweets - Save the vector embeddings of each tweet.

        m_example_tweets_given_bool - This important boolean will tell if only the example/dummy tweets are present. If that's
            the case the process of plotting will be handled a bit differently, more simple.



        m_max_length_of_tweets - When embedding occurs with the EmbedTweets, this will save the size because normally
            DisplayTweetEmbeddings will be run and then ShowPlotOfEmbeddedTweets right after. The last mentioned function
            will need the length to make embeddings.



        m_umap_description - Umap is a library used to get larger embeddings to a specified dimension, in this case, 2d.
            This is a brief description of it.

        m_tweet_and_points_description - This is describing a new dataframe with tweets and coordinates.

        m_k_means_description - Describing how k means works for this task.
    
    '''
    def __init__(self) -> None:
        self.m_model = None
        self.m_embedded_tweets = None
        self.m_example_tweets_given_bool = None
        self.m_list_of_tweets_to_predict = None
        self.m_plot_manager = ScatterPlot()
        self.m_max_length_of_tweets = 0


        self.m_umap_description = ''' Umap is used for dimension reduction. Some clustering algorithms can't handle vectors
        of really high dimensions and models typically put sentence/word embeddings **in** high dimensions. For example
        some models will have a 768 embedding vector and trying to visually see that is extremely difficult. With umap,
        this can be reduced to 2d so it can be much simplier for clustering algorithms to use. '''

        self.m_tweet_and_points_description = ''' Later, the coordinates will be displayed on the plot. To make it
        easier to map which point belongs to which tweet, this dataframe will display the coordinates next to their
        actual tweets. '''

        self.m_k_means_description = ''' The k means model will of course be using 3 clusters for positive, neutral, and
        negative. Its fit function will be called and given the x and y points created by umap in the previous step. Then
        it will provide the labels for each x & y pair, or to make things clearer, it will give predictions for every
        single tweet that was converted to numerical form and also reduced to 2d with umap. '''

        self.m_clustering_description = ''' Since this is an unsupervised learning problem, clustering algorithms will be
        applied so the data can be organized in some fashion. 3 clusters will be created, positive, neutral, and negative.
        Labels will be provided and those numbers will indicate which of the 3 clusters a particular tweet belongs too. '''

        self.m_tweets_from_twitter_tested = ''' **_With the new tweets pulled straight from twitter, these will of course
        be classified into one of the 3 labels like the example tweets._** '''
    


    def EmbedTweets(self, list_of_tweets, example_tweets_given_bool=False):
        # Most important first step is to embed the tweets, the model will do that.
        self.m_model = st.session_state.m_model 
        self.m_list_of_tweets_to_predict = list_of_tweets
        self.m_embedded_tweets = self.m_model.encode(list_of_tweets)
        self.m_max_length_of_tweets = len(self.m_embedded_tweets)

        self.m_example_tweets_given_bool = example_tweets_given_bool



    def DisplayEmbeddedTweets(self, num_of_tweets_to_display=5, values_in_each_tweet=10):
        st.write(f'1) Length of each tweet embedding: {len(self.m_embedded_tweets[0])}')
        st.write('')

        st.write(f'First {num_of_tweets_to_display} tweet embeddings. {values_in_each_tweet} displayed for each tweet.')
        for i in range(num_of_tweets_to_display):
            st.write(f'{i+1}) {self.m_embedded_tweets[i][:values_in_each_tweet]}')
            st.write('')



    def GetUmapReducedDimension(self):
        # Get some space so visuals on screen aren't so clustered up together.
        self.ApplySpacingOnScreen(4)
        st.write('#### Plot embedded tweets')

        # Use umap to change the embeddings.
        umap_obj = umap.UMAP(n_neighbors=self.m_max_length_of_tweets, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit(self.m_embedded_tweets)
        st.write('##### Use Umap to reduce dimensions of tweets.')
        st.write('_Description_')
        st.write(self.m_umap_description)
        st.write('')
        st.write('')
        st.write(umap_obj.embedding_)



        # Get some space so visuals on screen aren't so clustered up together.
        self.ApplySpacingOnScreen(5)

        # Create the dataframe to make displaying the future changes easier.
        st.write('##### Create new dataframe with x/y points for plot comparison.')
        st.write('_Description_')
        st.write(self.m_tweet_and_points_description)
        st.write('')
        st.write('')
        df = self.CreateEmbeddingsDataframe(umap_obj.embedding_, self.m_list_of_tweets_to_predict)
        st.write(df)

        return umap_obj, df








    '''
    
            Smaller helper functions.

    '''
    
    def GetEmbeddedTweets(self):
        return self.m_embedded_tweets


    # Help give spaces between portions of text.
    def ApplySpacingOnScreen(self, amount_of_spaces):
        for i in range(amount_of_spaces):
            st.write('')
    

    ''' This will be for creating the new dataframe that will match tweets with 2d points. First create the new dataframe with the given
        coordinates and make the columns x and y. Make a new column and give that the tweets. The x and y column and put first in front
        of the tweets so the new 3 lines of code reverses that so the tweets are first. This part doesn't necessarily matter too much, I
        just prefer the tweets first. Finally, return the dataframe. '''
    def CreateEmbeddingsDataframe(self, umap_embeddings_list, tweets_list):
        new_df = pd.DataFrame(umap_embeddings_list, columns=['x', 'y'])
        new_df['tweets'] = tweets_list

        columns = new_df.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        new_df = new_df[columns]

        return new_df