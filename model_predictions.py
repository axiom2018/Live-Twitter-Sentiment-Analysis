import streamlit as st
from data import Data


''' 

            Model building

The model that will be built will use pre trained word embeddings. This approach is better than training
a model from scratch because these models were trained on a seriously large amount of data and their
word embedding dimensions are finely tuned.
    
'''

# next_class added for chain of responsibility pattern.
class ModelPredictions(Data):
    def __init__(self):
        super().__init__()
        
        self.m_details = ''

        self.m_embedding_details = ''' With embeddings, words have a vector assigned to them. Think of 
        a 30 length vector of integers being given to the word "car" for example. This vector would be
        similiar to the vector for the word "truck", but not so similiar to say "economy". The next step
        is to get each matrix for the words in the glove embedding file. The size of the vectors was
        decided in the last step with the max_length variable. '''

        self.m_model_details = ''' The model used is capable of multiclassification with the labels neutral,
        positive, and negative. The model prediction will return a list of floats that match the models
        confidence regarding what category the tweet is in. Also the tokenizer that worked with the model
        is also saved in order to properly use text_to_sequences. '''

        # For the model and custom tokenizers to be loaded when the page is displayed.
        self.m_model = None



    def Display(self):
        st.session_state.m_model_class.Predict()

    
    
    def IsLastClass(self):
        return True