import streamlit as st
from data import Data
import pandas as pd
from models_strategy_pattern.model_types import ModelTypes


''' 

            Clustering

This is an unsupervised machine learning problem so to VISUALLY help the user with what's going on, showing
clustering algorithms would be great.
    
'''

# next_class added for chain of responsibility pattern.
class Clustering(Data):
    def __init__(self):
        self.m_details = ''' Since this project is an unsupervised machine learning problem, it'll be good to cluster similar 
        tweets together. Several clustering algorithms will be use such as the famous k means clustering algorithm. Also, example 
        or dummy tweets will be used in order to demonstrate the clustering since no tweets have been pulled yet. There will be 3 
        labels, positive, negative, and neutral. So all tweets will fall into one of those categories because getting the sentiment 
        of the public is very important to business.  '''

        self.m_final_details = ''' **In the next step, the model will be applied and turn the example tweets into embeddings!** '''

        self.m_example_tweets_details = ''' The following tweets contain made up tweets and others were found by going to Twitter
        and searching for "@CyberpunkGame" and clicking the "Latest" button to see the latest tweets that mention the page. With
        unsupervised problems, it helps if there's data graphed together despite not having labels because the tweets that will
        be pulled from Twitter have no labels. Several types of clustering methods will be done to compare and contrast.
        **_See list of tweets below_** '''

        # These tweets will be used for the example clusters.
        self.m_positive_tweets = ["Wonderful game, story was absolutely brilliant.",
                   "Wow! Simply gorgeous!",
                   "Having fun just walking around the city. I love how this game looks",
                   "Can't wait for the new DLC to come out, been playing the game for weeks like crazy and loving it.",
                   "I'd like to slow down time so I can spend more hours on this. #CyberpunkGame",

                   "Thank you so much !! I really appreciate that !!",
                   "Those are such great, moody captures!",
                   "Aside from all the negative bugs its a good game in general",
                   "A lot more to come on this one! Your game does a lot of things we'd love to see iterated upon. Rooting for it!",
                   "Loving it and can't wait for the new DLC. One of my favorite games for sure."]


        self.m_neutral_tweets = ["I guess.",
                  "Not bad, but not good.",
                  "It's okay I guess.",
                  "I mean it's alright.",
                  "Meh",

                  "Check out my latest vid Everyone Get Grenade's",
                  "What movie is this from?",
                  "An okayish game",
                  "A lot more to be desired but it's acceptable."]


        self.m_negative_tweets = ["This game was an absolute buggy piece of shit.",
                   "The developers literally released a product with thousands of bugs, turning the game into utter garbage. It's embarrassing.",
                   "The new game engine plans sounds terrible, somehow even worse than the game itself.",
                   '''😳😭  not sure about all the changes 🤔. It's making it hard for me carry on playing. To far in to restart. But also to many 
                   games to waste to much time gettin my head rd all the change 🙄.  Need to restart from beginning but again I got to many games to do that ''',
                   ''' Thank you for this update! I was really excited for it, but one of the bugs I’m experiencing isn’t fixed… My cell phone is 
                   red and I can’t make or receive phone calls so I’m still waiting for Takemura to call me back/call Judy and can’t. So I cannot progress in the game. ''',
                   
                   ''' why the hell did you make jackets rattle every step you take? Very weird to make loud rattling a feature of clothes, Cyberpunk is now a clothing
                    noise simulator? ''',
                   "The developers clearly don't care. They're releasing DLCs but SINCE inception this games major bugs weren't fixed, but patched over, so annoying.",
                   "I'd rather hacked games than this buggy piece of crap you guys made.",
                   "I'm sorry, the game crashed like 3 times in the hour I've played. 0 out of 5 stars.",
                   "Loading screen takes forever just to crash 2 minutes into gaming. I'm done with this.",
                   "Straight garbage"]


        ''' Also save the individual sets of tweets in session state. The reason for this is because in the algorithm
            to DISPLAY these tweets in a plot while they're of course separated by label, whether they're positive,
            neutral, or negative. The thing is the k means model in sklearn assigns it's OWN labels, and definitely
            won't follow mine. For example, in the below m_labels variable 0 = positive, 1 = neutral, 2 = positive.
            However the scikit learn k means will assign the positive tweets 2 instead. The problem becomes clear, 
            especially when one of the goals is to display color points representing the sentiment of each tweet.
            
            To solve this issue I'll have the original sets of tweets, then based on how k means decided to label each
            of them, separate those new sets. For example, if the positive tweets are labeled 2 by k means, put those in
            a list. Send that list into a function and compare it with all the original sets of tweets and see how many
            match based on each label. For example if the list has 8 tweets that match the original positive set, that's
            8 out of 10, so likely it's a positive set that the k means model labeled 2. But continue to loop through 
            others just to see if there might be a better match. 
            
            This way there will be consistency regarding which group that the k means model assigns to a set of tweets.
            Like the set will be confirmed as positive. '''

        if 'm_positive_tweets' not in st.session_state:
            st.session_state.m_positive_tweets = self.m_positive_tweets

        if 'm_neutral_tweets' not in st.session_state:
            st.session_state.m_neutral_tweets = self.m_neutral_tweets 
        
        if 'm_negative_tweets' not in st.session_state:
            st.session_state.m_negative_tweets = self.m_negative_tweets


        # Add the lists to get them all into one list before creating the dataframe for display purposes.
        self.m_all_example_tweets = self.m_positive_tweets + self.m_neutral_tweets + self.m_negative_tweets
        self.m_example_tweet_df = pd.DataFrame(data=self.m_all_example_tweets, columns=['Tweets'])

        # Get some labels in the dataframe too to mark the sentiment of the tweets. 0=positive, 1=neutral, 2=negative.
        self.m_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        self.m_example_tweet_df['labels'] = self.m_labels

        ''' Save the dataframe for display and comparison purposes in model_strategy_pattern subclasses. Whichever one may need it,
            like St_AllMpnetBaseV2.py for example. '''
        if 'm_example_tweet_df' not in st.session_state:
            st.session_state.m_example_tweet_df = self.m_example_tweet_df

        # Save in session state to be used in models to get embeddings.
        if 'm_example_tweets' not in st.session_state:
            st.session_state.m_example_tweets = self.m_all_example_tweets

        # Save the labels for use in models
        if 'm_labels' not in st.session_state:
            st.session_state.m_labels = self.m_labels



    def Display(self):
        st.title('Clustering')

        st.write('')
        st.write('')
        st.write(self.m_details)

        st.write('')
        st.write('')
        st.write('### 1) Example tweets')
        st.write(self.m_example_tweets_details)
        st.write('')
        
        st.write('##### a) Positive tweets')
        st.write(self.m_positive_tweets)
        st.write('')

        st.write('##### b) Neutral tweets')
        st.write(self.m_neutral_tweets)
        st.write('')

        st.write('##### c) Negative tweets')
        st.write(self.m_negative_tweets)
        st.write('')

        st.write('')
        st.write('')
        st.write('')
        st.write(f'##### {self.m_final_details}')

        # Signify when the user can move on.
        st.session_state.can_change_page = True



    def ModelCompatibilityCheck(self):
        model_class = st.session_state.m_model_class

        if self.ModelTypeCheck(model_class, self.__class__.__name__, ModelTypes.Embedding, True):
            return True

        # When the page is going to be skipped, increment variable that controls which class is displayed to get to the proper next class.
        st.session_state.list_index += 1
        print(f'Session state value in clustering.py: {st.session_state.list_index}')

        return False