import streamlit as st
from data import Data
# from models_strategy_pattern.custom_twitter_model import CustomTwitterModel
# from models_strategy_pattern.bert_model import BertModel
# from models_strategy_pattern.bert_base_uncased_model import BertBaseUncasedModel
from models_strategy_pattern.St_AllMpnetBaseV2 import St_AllMpnetBaseV2
from models_strategy_pattern.Vader import Vader


''' 

            Model selection

There are trained models with slight differences for this project that were created in Google Colab, and
there also is a BERT model available as well. BERT is a language model by Google so it can definitely 
help with nlp tasks. The user can decide here which model they'd like to use to begin the classification.

NONE of the models are trained while the project is being ran. The Google Colab models took some time
to train and BERT models take FOREVER to train. All were trained in a Google Colab, saved and are here
for reuse.
    
'''

class ModelSelection(Data):
    def __init__(self):

        ''' Variables:
        
        1) m_num_of_models_to_choose_from - This project might be updated at times to include newer
            or just slightly edited models. This variable will be used in this class to keep the user
            up to date on how many models are available. '''
        self.m_num_of_models_to_choose_from = 1 # Was 2.

        self.m_details = f''' The world of machine learning has multiple types of models for multiple types
        of tasks. There are {self.m_num_of_models_to_choose_from} models to choose from and they are explained
        below. Each model will show it's summary after predictions are made. '''

        # The types of models to choose from.
        self.m_model_types = ['SentenceTransformer (all-mpnet-base-v2)',
                              'Vader']

        # Holds the specific chosen model.
        self.m_model_class = None


    def ModelCompatibilityCheck(self):
        return True


    def Display(self):
        st.title('Model Selection')
        st.write(self.m_details)
        st.write('')
        st.write('')
        st.write('')

        st.write('#### Select the type of model to use below!')
        st.write('')
        st.write('')

        # Placeholder for the select box, after button is pressed, it's no longer needed.
        place_holder_for_select_box = st.empty()

        # Use placeholder for a temporary button that needs only 1 click.
        place_holder_for_button = st.empty()

        # type_of_model = st.selectbox(f'Select type of model to use: ', self.m_model_types)
        type_of_model = place_holder_for_select_box.selectbox(f'Select type of model to use: ', self.m_model_types)

        if place_holder_for_button.button('Select Model') is True:
            # Get rid of the select box and button.
            place_holder_for_select_box.empty()
            place_holder_for_button.empty()

            st.write(f'Type of model chosen is {type_of_model}')

            ''' Now save the model here. The interface is already set in model_strategy.py so any functions that
                are called with this object from here on in only need to adhere to those interface function names.
                
                Implement factory pattern later here. '''
            if type_of_model == self.m_model_types[0]:
                self.m_model_class = St_AllMpnetBaseV2()
            elif type_of_model == self.m_model_types[1]:
                self.m_model_class = Vader()

            # Use the spinner because no matter what, loading will take some time.
            with st.spinner(f'Loading {self.m_model_class.GetName()} into memory....'):
                self.m_model_class.Load()

            st.write(f'{self.m_model_class.GetName()} model loaded!')

            # Save the model CLASS that CONTAINS the model.
            if 'model_class' not in st.session_state:
                st.session_state.m_model_class = self.m_model_class

            # The model is chosen, now the user can proceed to the next page.
            st.session_state.can_change_page = True