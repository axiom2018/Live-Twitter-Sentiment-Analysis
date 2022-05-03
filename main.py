import streamlit as st
from intro_page import IntroPage
from model_selection import ModelSelection
from get_tweets import GetTweets
from preprocess_tweets import PreprocessTweets
from model_predictions import ModelPredictions
from get_tweet_embeddings import GetTweetEmbeddings
from clustering import Clustering
from unsupervised_learning import UnsupervisedLearning
from streamlit import legacy_caching
from streamlit_autorefresh import st_autorefresh


# Clears the cache and re-runs the code. Helps for refreshing properly.
legacy_caching.clear_cache()


# A list will be made of class objects so the user will navigate step by step to the end. This value will keep track of what class to display.
if 'list_index' not in st.session_state:
    st.session_state.list_index = 0


# Boolean to ensure the user can't hit next without doing the task the page requires. Ex: Selecting a model, getting tweets, etc.
if 'can_change_page' not in st.session_state:
    st.session_state.can_change_page = False


# 1 - IntroPage - Simple introduction to user regarding what this project will demonstrate.
#
# 
# 2 - ModelSelection - Choose a model to use for the project. You can always  refresh the project with F5 and choose again.
#
#
# 3 - UnsupervisedLearning - Quick description on what unsupervised learning IS because this project is definitely unsupervised. Fresh tweets will be
#       pulled straight from Twitter and the sentiment will be extracted from them in multiple ways.
#
#
# 4 - Clustering - The idea of clustering is grouping similar things together. This will be a necessary step in this project, but not for ALL types of
#       models.
#
#
# 5 - GetTweetEmbeddings - Embeddings can be made of sentences/tweets, and then transformed to be used for clustering on a 2d graph.
#
#
# 6 - GetTweets - Get a certain amount of tweets from the target page for the user.
#
# 
# 7 - ProcessTweets - Raw text of course cannot be understood by the model. It must be processed to remove things like stop words and other punctuation 
#       related things.
#
#
# 8 - ModelPredictions - Whichever model was selected, this is the time where the predictions will occur.


# From feature engineering and onward, ONLY use the updatedDf in session_state.
if 'list_of_classes' not in st.session_state:
    # Each class will be in a list.
    st.session_state.list_of_classes = [IntroPage(),
                                        ModelSelection(),
                                        UnsupervisedLearning(),
                                        Clustering(),
                                        GetTweetEmbeddings(),
                                        GetTweets(),
                                        PreprocessTweets(),
                                        ModelPredictions()
                                        ]


# Page navigation button.
_ ,next = st.columns([10, 1])


# To ensure the button dissapears when the user reaches the last page, use an empty place holder.
place = st.empty()


# If this specific class IS the last class, only worry about displaying, there's no need to worry about a "next" button.
if st.session_state.list_of_classes[st.session_state.list_index].IsLastClass():
    st.session_state.list_of_classes[st.session_state.list_index].Display()
else:
    # When the next button is pressed, increase index value and the Display function show the next classes material.
    if place.button('Next') and st.session_state.can_change_page is True:
        st.session_state.list_index += 1

        # If the last class is reached, remove the next button.
        if st.session_state.list_index >= len(st.session_state.list_of_classes):
            place.empty()

        # Reset boolean to start process over.
        st.session_state.can_change_page = False


    if st.session_state.list_of_classes[st.session_state.list_index].ModelCompatibilityCheck():
        # All classes are derived/sub classes of the base class Data, which implements the Display function incase the streamlit approach is used.
        st.session_state.list_of_classes[st.session_state.list_index].Display()
    else:
        # Auto refreshing the page is necessary. For example, if the St_AllMpnetBaseV2.py model has model type "embedding"
        # and the UnsupervisedLearning page is of type "SimpleLibrary", that means that the afforementioned page will be skipped
        # when this particular model is selected. This is controlled by the ModelCompatibilityCheck function and if it returns 
        # false, beforehand it will increase the list_index value. Basically telling the code here to go ahead and try checking
        # the NEXT class to see if that one is a match.
        count = st_autorefresh(interval=1000, limit=100, key="refresh_the_page")