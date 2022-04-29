import streamlit as st
from data import Data
from models_strategy_pattern.model_types import ModelTypes

''' 
            Get tweet embeddings

A very crucial step is to be able to get sentence embeddings in this project. This is important because
this project is a big unsupervised problem as it is. Being able to get texts and label them as positive,
negative, or neutral. Clustering tweets will also be a thing, and that of course revolves around embeddings
as well.
    
'''

# next_class added for chain of responsibility pattern.
class GetTweetEmbeddings(Data):
    def __init__(self):        
        self.m_details = ''' To understand sentence embeddings, it's worth going over **_word_** embeddings as well. A word
        embedding is basically a vector or numerical representation of a single word and is able to identify the syntaxes.
        Some famous word embedding libraries are Word2Vec, and GloVe, but there's a few more out there. Sentence embeddings
        are basically the same thing except with of course sentences. With sentence embeddings, it's possible to retain context
        in which a certain word was used IN the sentence. '''

        self.m_conversion_details = ''' Embeddings for words and sentences are long vectors/arrays/lists of different sizes. A 
        typical size for an embedding is 768, especially with more popular models. The idea is that any sentence (or tweet in
        this case) can be a **long** vector of floating point values. This has to be done of course since models don't understand
        text but only numbers. '''

    

    def Display(self):
        print('get_tweet_embeddings.py check 1.')
        st.title('Get Tweets Embeddings')

        st.write('')
        st.write('')
        st.write(self.m_details)

        st.write('')
        st.write('')
        st.write('')
        st.write('### Convert the tweets into embeddings')
        st.write(self.m_conversion_details)
        
        st.write('')
        st.write('')
        st.write('')
        st.write('### Display the embedded tweets')

        st.session_state.m_model_class.Predict() # Argument is a list.

        # The model is chosen, now the user can proceed to the next page.
        st.session_state.can_change_page = True
        print('get_tweet_embeddings.py check 2.')



    def ModelCompatibilityCheck(self):
        model_class = st.session_state.m_model_class

        if self.ModelTypeCheck(model_class, self.__class__.__name__, ModelTypes.Embedding, True):
            return True

        # When the page is going to be skipped, increment variable that controls which class is displayed to get to the proper next class.
        st.session_state.list_index += 1
        
        return False