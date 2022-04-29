import streamlit as st
from Etc.Embeddings import Embeddings
from Etc.ScatterPlot import ScatterPlot
from Etc.Spacing import Spacing
from models_strategy_pattern.model_strategy import ModelStrategy
from models_strategy_pattern.model_types import ModelTypes
from sentence_transformers import SentenceTransformer
from clustering_strategy_pattern.k_means import SKL_KMeans




''' 
            St_AllMpnetBaseV2

    This is a model that the sentence transformers framework utilizes, makes the process of getting sentence 
    embeddings

'''

class St_AllMpnetBaseV2(ModelStrategy, Spacing):
    def __init__(self):
        super().__init__(ModelTypes.Embedding, "All Mpnet Base V2")
        
        self.m_model_details = ''' This model is one of the few pretrained models provided by sentence bert (or sbert).
        All of the models for sentence bert have been assessed for how well they perform and this one inparticular has
        the **_best_** quality. '''

        self.m_tweet_and_points_description = ''' Later, the coordinates will be displayed on the plot. To make it
        easier to map which point belongs to which tweet, this dataframe will display the coordinates next to their
        actual tweets. '''

        self.m_new_predictions_description = ''' The prediction system here will be a bit different from probably what's expected.
        In most unsupervised projects, clustering algorithms are used and rightfully so. However there _must_ be some given data
        in order **to** get some beginning clusters. With that said, the example tweets from before will be the main ones that
        help create clusters. However the _new_ tweets will "decide" which group they belong in with simple math. The 3 groups of
        tweets, positive, neutral, and negative will have their tweets converted into embeddings and down to smaller 2d dimensions
        with dimension reduction methods such as umap. All 3 groups will have their embeddings added and the mean found. Handling
        new tweets and deciding what group _they_ belong in, it's simply a matter of finding out which mean of which group that
        this new tweets 2d points relates to more.  '''

        self.m_new_df_description = ''' This dataframe will be used mainly for classifying the new tweets pulled straight from Twitter
        in a different manner from what was done before. The old tweets are in the dataframe and are classified as positive, neutral,
        or negative. With the x & y points now found with each _old_ tweet, go ahead and find the mean of all the groups. For example,
        looking at the above dataframe, get the mean of the tweets with the label of 0. The same applies to the other groups. Then
        with a **_NEW_** tweet pulled straight from Twitter, get it's x and y coordinates and see what group is closer to it and voila!
        **_This may or may not produce different results from standard clustering algorithms like k means, but it's worth a shot!_** '''
        
        self.m_embeddings_manager = Embeddings()
        self.m_clustering_algorithm = SKL_KMeans(1234)
        self.m_plot_manager = ScatterPlot()



    @st.cache(allow_output_mutation=True)
    def Load(self):
        if 'm_model' not in st.session_state:
            st.session_state.m_model = SentenceTransformer('all-mpnet-base-v2')
            


    def Predict(self, num_of_tweets_to_display=5, values_in_each_tweet=10):
        st.title(f'{self.m_model_name} predictions.')
        self.ApplySpacingOnScreen(3)
        
        # Example tweets were created in clustering.py, will be needed to get total tweets.
        example_tweets_list = st.session_state.m_example_tweets

        ''' Get the necessary dataframe created in "get_tweets.py" file and get the tweets from it to pass to the function below. Convert the
            tweets to a list and the "Cleaned Tweets" column was added in the "preprocess_tweets.py" file. '''
        new_tweets_from_twitter_list = None
        
        if 'dataframe_of_tweets' in st.session_state:
            df = st.session_state.dataframe_of_tweets
            new_tweets_from_twitter_list = df['Cleaned Tweets'].tolist()


        total_tweets = None

        if new_tweets_from_twitter_list is None:
            # If none, this is being called for the purpose of displaying example tweets only.
            total_tweets = example_tweets_list
        else:
            # If not none, The new tweets will be combined with the old tweets to ensure they have a good starting point.
            total_tweets = example_tweets_list + new_tweets_from_twitter_list


        # Get the tweets embedded by the model loaded earlier and display them.
        self.m_embeddings_manager.EmbedTweets(total_tweets)
        self.m_embeddings_manager.DisplayEmbeddedTweets(num_of_tweets_to_display, values_in_each_tweet)

        # Reduce the embeddings of course. Function returns a umap object and dataframe with the tweets and new x and y points.
        umap_obj, df = self.m_embeddings_manager.GetUmapReducedDimension()

        # Fit will display details about clustering, and more.
        self.ApplySpacingOnScreen(5)
        self.m_clustering_algorithm.Fit(umap_obj.embedding_)

        # Add the labels to the dataframe.
        self.ApplySpacingOnScreen(5)
        st.write('##### Update the previously created dataframe with clustering algorithm labels')
        df['labels'] = self.m_clustering_algorithm.GetLabels()
        st.write(df)

        # Now plot.
        self.ApplySpacingOnScreen(5)
        self.m_plot_manager.Plot(df, 'labels', 'tweets', (9, 9))

        # Give a description about the plot.
        self.ApplySpacingOnScreen(5)
        self.m_plot_manager.PlotDetailedDescription(self.m_clustering_algorithm.GetLabelsAsList())