from sklearn import model_selection
import streamlit as st
from intro_page import IntroPage
from model_selection import ModelSelection
from get_tweets import GetTweets
from preprocess_tweets import PreprocessTweets
from tokenization import Tokenization
from model_predictions import ModelPredictions
from get_tweet_embeddings import GetTweetEmbeddings
from clustering import Clustering
from unsupervised_learning import UnsupervisedLearning
from streamlit import legacy_caching
from streamlit_autorefresh import st_autorefresh


legacy_caching.clear_cache()


# A list will be made of class objects so the user will navigate step by step to the end.
if 'list_index' not in st.session_state:
    st.session_state.list_index = 0


# Boolean to ensure the user can't hit next without doing the task the page requires. Ex: Selecting a model, getting tweets, etc.
if 'can_change_page' not in st.session_state:
    st.session_state.can_change_page = False


# 1 - IntroPage - Simple introduction to user regarding what this project will
#       demonstrate.
#
#
# 2 - GetTweets - Get a certain amount of tweets from the target page as
#       requested by the user.
#
# 
# 3 - ProcessTweets - Raw text of course cannot be understood by the model.
#       It must be processed to remove things like stop words and other 
#       punctuation related things.
#
#
# 4 - Tokenization - Another preprocessing step, tokenization will take the words
#       and replace them with numbers so the model udnerstands.
#
#
# 5 - ModelBuilding - This will create/load a model which will allow the user to
#       see predictions.
#
#
#



# From feature engineering and onward, ONLY use the updatedDf in session_state.
if 'list_of_classes' not in st.session_state:
    # Each class will be in a list.
    st.session_state.list_of_classes = [ModelSelection(),
                                        UnsupervisedLearning(),
                                        Clustering(),
                                        GetTweetEmbeddings(),
                                        GetTweets(),
                                        PreprocessTweets(),
                                        Tokenization(),
                                        ModelPredictions()
                                        # IntroPage()
                                        ]


# Page navigation button.
_ ,next = st.columns([10, 1])


# To ensure the button dissapears when the user reaches the last page, use an empty place holder.
place = st.empty()



if st.session_state.list_of_classes[st.session_state.list_index].IsLastClass():
    st.session_state.list_of_classes[st.session_state.list_index].Display()
else:
    # When the next button is pressed, increase index value and the Display function show the next classes material.
    if place.button('Next') and st.session_state.can_change_page is True:
        st.session_state.list_index += 1

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