import streamlit as st
import regex as re
from data import Data
import nltk
from nltk.corpus import stopwords

from models_strategy_pattern.model_types import ModelTypes


''' 

            PreprocessTweets

Last step was GATHERING tweets, this step will be used to preprocess them. 
    
'''

# next_class added for chain of responsibility pattern.
class PreprocessTweets(Data):
    def __init__(self):

        self.m_details = ''' Preprocessing is a necessary step when dealing with **any** sentiment analysis task and depending on the 
        task itself, there are a variety of steps. For example, in Twitter it's a lot more common to see "@" mentions of other usernames than on
        Facebook. Also the hashtags, retweets, links and such must be handled in this social media platform. '''

        # Easily pre process stop words when the time comes.
        nltk.download('stopwords')
        if 'stop_words' not in st.session_state:
            st.session_state.stop_words = set(stopwords.words('english'))
    


    def Display(self):
        st.title('Preprocess tweets')
        st.write('')
        st.write('')
        st.write(self.m_details)

        st.write('')
        st.write('')
        st.write('#### 1) Before preprocessing the tweets: ')

        # Show the specified amount of tweets in detail with full text and screen name.
        num_of_tweets = st.session_state.num_of_tweets_to_display

        for i in range(num_of_tweets):
            tweet = st.session_state.list_of_tweets[i].full_text
            user = st.session_state.list_of_tweets[i].user.screen_name
            st.write(f'{i+1}) {tweet}')
            st.write(f'---By user: {user}')


        self.PreprocessTweets()


        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('#### 3) After preprocessing: ')

        # Display all post preprocessed tweets with dataframe.
        df = st.session_state.dataframe_of_tweets

        for i in range(num_of_tweets):
            tweet = df['Cleaned Tweets'][i]
            st.write(f'{i+1}) {tweet}')

        # Also update the dataframe since the cleaning is done.
        st.session_state.dataframe_of_tweets = df


        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(f'#### 4) Updated dataframe: ')
        st.write(st.session_state.dataframe_of_tweets)

        # Tweets are preprocessed, now the user can proceed to the next page.
        st.session_state.can_change_page = True



    def ModelCompatibilityCheck(self):
        model_class = st.session_state.m_model_class

        if self.ModelTypeCheck(model_class, self.__class__.__name__, ModelTypes.Embedding, True):
            return True

        # When the page is going to be skipped, increment variable that controls which class is displayed to get to the proper next class.
        st.session_state.list_index += 1
        print(f'Session state value in preprocess_tweets.py: {st.session_state.list_index}')

        return False


    
    '''
    
        Slight edits to the old preprocessing. I'd like to see if the most natural state of the tweet can still yield
        decent or good results. So I'll eliminate steps such as punctuation since in the clustering step the tweets
        kept their punctuation.

        These changes will be denoted with the comment "#te" meaning [t]emporarily [e]dited.
    
    '''
    def PreprocessTweets(self):
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('#### 2) What tweets will have removed from them: ')
        st.write('a) @ mentions')
        st.write('b) # hashtags')
        st.write('c) Retweets')
        st.write('d) Links')
        # st.write('e) Punctuation') #te

        df = st.session_state.dataframe_of_tweets

        #   Remove mentions, hashtags, etc
        df['Cleaned Tweets'] = df['Tweets'].apply(self.ProcessTweets)

        #   Lower the text.
        # df['Cleaned Tweets'] = [tweet.lower() for tweet in df['Cleaned Tweets']] #te

        # Finally update the dataframe since the cleaning is done.
        st.session_state.dataframe_of_tweets = df

        # # Display the number of clean tweets in numerical order.
        # for i in range(num_of_cleaned_tweets):
        #     tweet = st.session_state.dataframe_of_tweets['Cleaned Tweets'][i]
        #     st.write(f'{i+1}) {tweet}')


        #te (below)

        # st.write('')
        # st.write('')
        # # Get rid of stop words.
        # df = st.session_state.dataframe_of_tweets
        # df['Cleaned Tweets'] = df['Cleaned Tweets'].apply(self.CleanStopWords) 



    ''' Cleaning text has quite a few steps to it and of course changing the steps
        can alter the outcome.

        The 'r' makes it clear that the expression is a raw string, or the pattern
        itself. Once found, just replace it with literally nothing, essentially 
        getting rid of it.
        
        In order, the function will remove the following things:

        1) @ mentions.
        2) # symbol.
        3) Retweets. 
        4) Links.
        5) Punctuation. '''
    def ProcessTweets(self, text):
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
        text = re.sub(r'#(\w+)', '', text)
        text = re.sub(r'RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/\S+', '', text)
        # text = re.sub(r'[^\w\s]', '', text) #te

        return text



    ''' .split - Takes string and turns it into a list.
    
        List comprehension - Get every word in the list and if it's NOT found in stopwords,
            keep it.
            
        .join - Combine the list into one, space separated.  '''
    def CleanStopWords(self, text):
        words_in_text = text.split()
        words_in_text = [w for w in words_in_text if not w in st.session_state.stop_words]
        return ' '.join(words_in_text)