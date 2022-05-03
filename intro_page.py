import streamlit as st
from data import Data
from PIL import Image

''' 

            Intro

This will just be an introduction for the user on what the project is about.
    
'''

class IntroPage(Data):
    def __init__(self):
        super().__init__()
        
        # Fancy the intro page up a bit with a picture.
        self.m_image = Image.open('images/sentiment.png')

        # Full intro message for the user on what project is about.
        self.m_intro_message = ''' This project will pull tweets from Twitter and attempt to analyze their sentiment using pretrained word embeddings. What **_are_** 
        word embeddings? They're a way to capture a relationship between words while converting words to numbers since models do not understand text. This is much 
        better than simply encoding words into integers since that will give no relationship. Also it's **definitely** better than one hot encoding because that 
        approach will generate a new column for every word, exploding the dataset and **still** will not get a proper relationship. '''

        self.m_etc = "Let's begin!"

    
    def Display(self):
        st.title('_Live Twitter Sentiment Analysis_')

        st.image(self.m_image, caption='Sentiment meter.')

        st.write(self.m_intro_message)
        st.write(self.m_etc)

        # Set to true so the next button becomes functional.
        st.session_state.can_change_page = True