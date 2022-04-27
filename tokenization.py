import streamlit as st
from data import Data
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models_strategy_pattern.model_types import ModelTypes


''' 

            Tokenization

This can be said to be part 2 of preprocessing. Didn't want to clutter the PreprocessTweets class.
With that said this will handle tokenization which will convert words to numbers.
    
'''

class Tokenization(Data):
    def __init__(self):
        super().__init__()
        
        # Message for the specific page.
        self.m_details = ''' Models aren't advanced enough yet to understand raw text. So tokenization will
        help convert the words to numbers. And this method will in fact give a relationship to words so 
        that's a plus. '''

        self.m_tokenization_details = ''' If there's sentences like 'I love my dog' & 'I love my cat', then 
        they will be converted in a manner like [1,2,3,4] & [1,2,3,5]. Then after fit on texts which basically 
        pairs the words with numbers we can get a word index like {'i':4, 'love':2, 'my':1, 'dog':3, 'cat':5}. 
        So the full result would look like: [[4,2,1,3], [4,2,1,5]]. '''

        # self.m_max_length_details = ''' The max length will be used for padding but it can have a significant 
        # effect on the outcome. '''
    
        # self.m_padding_details = f''' The model requires that the text be the same size. If the size must be 
        # 50 but the length of a tweet is 36, padding will force it up to 50 by adding 0's. If the length of a
        # tweet is 72, padding will truncate and cut it down to 50. A bit of information loss occurs with this
        # version though. '''



    def Display(self):
        st.title('Tokenization')
        st.write(self.m_details)
        st.write('')
        st.write('')
        st.write('')

        # Show all the tweets because they'll be tokenized soon.
        df = st.session_state.dataframe_of_tweets
        st.write(f'### Tweets to be tokenized are the following: ')
        st.write(df['Cleaned Tweets'])
        st.write('')
        st.write('')
        st.write('')

        # Give an explanation to the user. They may or may not be familiar with the process.
        st.write(f'### Explanation of how Tokenization works:')
        st.write(self.m_tokenization_details)


        ''' [Referring to saved model] 
        
            Begin tokenization then show tokenized tweets '''
        st.session_state.m_model.Tokenize()
        st.session_state.m_model.DisplayTokenizedTweets()


        ''' Now for padding. Clearly all the tweets are a different size so that's not going
            to fly with the model. Padding gives the ability to add more to a shorter tweets.
            If the tweet is long, truncation is possible which is chopping down the size.  '''
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(f'### Explanation of padding details: ')

        ''' [Referring to saved model] 
        
            Show padding details. '''
        st.session_state.m_model.DisplayPaddingDetails()

        # Tokenization done, now the user can proceed to the next page.
        st.session_state.can_change_page = True


    def ModelCompatibilityCheck(self):
        # model_class = st.session_state.m_model_class

        # if self.ModelTypeCheck(model_class, self.__class__.__name__, ModelTypes.SimpleLibrary, True):
        #     return True

        # When the page is going to be skipped, increment variable that controls which class is displayed to get to the proper next class.
        st.session_state.list_index += 1
        print(f'NO check here. Will always return false. Session state value in tokenization.py: {st.session_state.list_index}')

        return False