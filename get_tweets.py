import pandas as pd
import streamlit as st
from data import Data
import tweepy



''' 

            GetTweets

This class will be solely responsible for getting tweets from Twitter. Preprocessing them will be another step. 
A lot of keys (strings basically) have to be known in order to be able to pull tweets right from twitter. 

    Currently the code only searches for Cyberpunk tweets, but it's possible to have the user search for their
    own desired @. That might be implemented later.
    
'''

# next_class added for chain of responsibility pattern.
class GetTweets(Data):
    def __init__(self):
        
        # Also called consumer key.
        self.m_api_key = '8qpz7Nw6SX6MYmUYMvGQJXz7O'

        # Also called consumer secret.
        self.m_api_secret = '03uf9oRUR8IQ7NN5at0JmgGzraRvfSmY2wmgOtYkb2YoJNmk9t'

        # More credentials to get tweets.
        self.m_access_token = '1394821284249227264-4GRyzqqHqVVxqmpyR88bGGQ72DriUG'
        self.m_access_token_secret = 'Jexnu5amTnUKGTB4hK7b0gZTY4vs5yOmishGf3lrlGzjR'

        # Begin authentication 
        self.m_auth_handler = tweepy.OAuthHandler(self.m_api_key, self.m_api_secret)
        self.m_auth_handler.set_access_token(self.m_access_token, self.m_access_token_secret)
        self.m_api = tweepy.API(self.m_auth_handler)

        # There will be a limit regarding how many tweets the user request.
        self.m_tweet_amounts = [30, 50, 70]

        # Message for the specific page.
        self.m_details = ''' Get the amount of tweets requested via the drop box below. After
        that a dataframe will be created out of the tweets since data scientists are familiar
        with dataframes. '''

        self.m_dataframe_details = ''' A dataframe is essientially a 2d array with many features (or columns)
        and each row is a portion of data. They are of course normally used in machine learning so the tweets
        will be put a dataframe of their own. '''

        # See the mentions on twitter of the specified account, and how many to display.
        if 'account_to_search' not in st.session_state:
            st.session_state.account_to_search = 'CyberpunkGame'

        if 'num_of_tweets_to_display' not in st.session_state:
            st.session_state.num_of_tweets_to_display = 5



    def Display(self):
        st.write(f'## Now to get {st.session_state.account_to_search} tweets.')
        st.write(self.m_details)
        st.write('')
        st.write('')
        st.write('')

        # Use placeholder for a temporary button that needs only 1 click.
        place = st.empty()

        # Make sure to get desired value of tweets with this box.
        amount_of_tweets_to_get = st.selectbox(f'Select amount of tweets to get:', self.m_tweet_amounts)

        if place.button('Get Tweets') is True:
            # Get rid of the button.
            place.empty()

            # Save the amount of tweets for later use in tokenization.py.
            if 'amount_of_tweets_to_get' not in st.session_state:
                st.session_state.amount_of_tweets_to_get = amount_of_tweets_to_get

            ''' Begin collecting the tweets.
                
                filter:retweets - Be sure to not get any retweets, those are just tweets of people that reply to eachother.
                lang - en for English but it's possible to do other languages if the model supports it.  '''
            tweets = tweepy.Cursor(self.m_api.search, q='CyberpunkGame -filter:retweets', lang='en', tweet_mode='extended').items(amount_of_tweets_to_get)

            # Save them in a session state list so it can be carried to the next page.
            if 'list_of_tweets' not in st.session_state:
                st.session_state.list_of_tweets = []

                for tweet in tweets:
                    st.session_state.list_of_tweets.append(tweet)

            num_of_tweets = st.session_state.num_of_tweets_to_display
    
            st.write('')
            st.write('')
            st.write('')
            st.write(f'## Displaying {num_of_tweets} most recent tweets.')
            for i in range(num_of_tweets):
                tweet = st.session_state.list_of_tweets[i].full_text
                user = st.session_state.list_of_tweets[i].user.screen_name
                st.write(f'{i+1}) Tweet: {tweet}\nUser: {user}')


            self.CreateDataframeFromTweets()

            # The tweets are there, now the user can proceed to the next page.
            st.session_state.can_change_page = True

    

    ''' As the name suggests, this functon will get all tweets received in the
        display function and just toss them into a dataframe. This is done because
        a lot of operations are normally used on dataframes so there's a strong
        familarity there. '''
    def CreateDataframeFromTweets(self):
        if 'dataframe_of_tweets' not in st.session_state:
            # Make code more readable by getting the tweets saved in streamlit
            list_of_tweets = st.session_state.list_of_tweets

            # Use list comprehension to set up the dataframe.
            df = pd.DataFrame([tweet.full_text for tweet in list_of_tweets], columns=['Tweets'])

            # Save in the session state.
            st.session_state.dataframe_of_tweets = df

            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('## Dataframe of tweets: ')
            st.write(df)