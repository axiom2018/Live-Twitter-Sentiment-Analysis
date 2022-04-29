import streamlit as st
from data import Data
from PIL import Image
from models_strategy_pattern.model_types import ModelTypes

'''

            Unsupervised learning

The goal of the project is to analyze sentiment from unlabeled data retrieved from Twitter, so this project is definitely
centered around unsupervised learning.


'''

class UnsupervisedLearning(Data):
    def __init__(self):        
        self.m_image = Image.open('images/unsupervised_machine_learning.png')

        self.m_details = ''' What is unsupervised learning? This form of machine learning uses unlabeled data and tries to
        make sense of that. Clustering unlabeled data will give a deeper understanding of what data is more related than others.
        There are many clustering algorithms to do this. The main point behind unsupervised machine learning is it's much more 
        independent than supervised learning, which **requires** labeled data so the model can have data that already makes 
        sense. '''


    
    def Display(self):
        st.title('**What** is Unsupervised Learning?')

        st.image(self.m_image, caption='Unsupervised machine learning.')

        st.write('')
        st.write('')
        st.write(self.m_details)

        st.session_state.can_change_page = True



    def ModelCompatibilityCheck(self):
        model_class = st.session_state.m_model_class

        if self.ModelTypeCheck(model_class, self.__class__.__name__, ModelTypes.SimpleLibrary, True):
            return True

        # When the page is going to be skipped, increment variable that controls which class is displayed to get to the proper next class.
        st.session_state.list_index += 1
        print(f'Session state value in unsupervised_learning.py: {st.session_state.list_index}')

        return False