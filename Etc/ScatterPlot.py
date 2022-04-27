import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

''' 
            ScatterPlot

This class will take on all scatter plot tasks the models will have. 

'''

class ScatterPlot:
    def __init__(self):
        self.m_zero_tweets = None
        self.m_one_tweets = None
        self.m_two_tweets = None

        self.m_colors = None

        self.m_post_plot_description_one = ''' Now there are a lot of different colored points in the plot. What does their color
        represent? See below colored text for answer. '''

        self.m_post_plot_description_two = ''' So **_why_** are the points in the plot colored this way? Well in step 3 the clustering algorithm 
        went ahead and fit itself on the 2d points brought to us by umap (in step 1) and gave labels to each tweet. The labels are displayed
        above. We're guaranteed 3 values which are 0, 1, and 2, because that was the amount of clusters specified _to_ the clustering algorithm.
        Only 3 clusters necessary since positive, neutral, and negative are the only labels in this project. The labels that the clustering 
        algorithm given for each tweet, as shown earlier, are in order with the tweets in the dataframe. '''

        self.m_post_plot_description_three = ''' The tweets are split up by the labels 0, 1 and 2. Remember in the Clustering page there were 3
        sets of example or dummy tweets created. Those original 3 sets are compared to the new split up sets of tweets that were split
        by their labels 0, 1, and 2. This is done to see what the new set is more like overall. For example, if a new set of tweets
        had 5 maximum tweets, 4 are positive, and 1 is negative. That means that this new set is more positive than neutral or negative.
        The function that does this returns one of the colors displayed in the colored text above. **_Below shows separated tweets by label._**  '''


    ''' Arguments:

        1) df - The dataframe to pass in. The project is built around a 3 sentiment system of labels 0, 1, and 2. So the dataframe
                    will be used to take tweet information from it.

        2) labels_column_name/tweets_column_name - labels_column_name will be of course whatever the labels column is called. The 
                    system of course looks for 0, 1 and 2 as labels. Then the tweets_column_name is whatever that column is called.
                    To make sure the code will still work despite potential name changes, that's why these arguments are provided.

        3) colors - These can be provided

    '''
    def Plot(self, df, labels_column_name, tweets_column_name, plot_size, example_tweets_given_bool=False):
        st.write('##### Organized points and display plot.')

        # Begin using the dataframe to split up the tweets.
        self.m_zero_tweets = df[df[labels_column_name] == 0][tweets_column_name].tolist()
        self.m_one_tweets = df[df[labels_column_name] == 1][tweets_column_name].tolist()
        self.m_two_tweets = df[df[labels_column_name] == 2][tweets_column_name].tolist()


        # Run the function to get the colors for each group.
        self.m_colors = []

        # if example_tweets_given_bool is True:
        #     self.m_colors.append(self.GetLabelForTweets(self.m_zero_tweets))
        #     self.m_colors.append(self.GetLabelForTweets(self.m_one_tweets))
        #     self.m_colors.append(self.GetLabelForTweets(self.m_two_tweets))
        # else:

        self.m_colors.append(self.GetLabelForTweets(self.m_zero_tweets))
        self.m_colors.append(self.GetLabelForTweets(self.m_one_tweets))
        self.m_colors.append(self.GetLabelForTweets(self.m_two_tweets))

    
        # Use the dataframe to get all points according to their labels.
        zero_x_points = df[df['labels'] == 0]['x'].tolist()
        zero_y_points = df[df['labels'] == 0]['y'].tolist()

        one_x_points = df[df['labels'] == 1]['x'].tolist()
        one_y_points = df[df['labels'] == 1]['y'].tolist()

        two_x_points = df[df['labels'] == 2]['x'].tolist()
        two_y_points = df[df['labels'] == 2]['y'].tolist()


        # Multiple scatter plots are need for all the different points so plt.subplots is used.
        fig, ax = plt.subplots(figsize=plot_size)

        plt.scatter(zero_x_points, zero_y_points, color=self.m_colors[0][1], s=20, cmap='Spectral')
        plt.scatter(one_x_points, one_y_points, color=self.m_colors[1][1], s=20, cmap='Spectral')
        plt.scatter(two_x_points, two_y_points, color=self.m_colors[2][1], s=20, cmap='Spectral')

        st.plotly_chart(fig)

    

    def PlotDetailedDescription(self, list_of_labels):
        st.write('##### Understanding the plot.')
        st.write('_Description_')
        st.write('')

        # This is streamlit unsafe html, but it's necessary to display colors to get the users to understand clearly what's happening.
        color_message_one = self.GetColoredText(self.m_colors[0][1], self.m_colors[0][0])
        color_message_two = self.GetColoredText(self.m_colors[1][1], self.m_colors[1][0])
        color_message_three = self.GetColoredText(self.m_colors[2][1], self.m_colors[2][0])

        # Brief explanation.
        st.write(self.m_post_plot_description_one)

        st.markdown(color_message_one, unsafe_allow_html=True)
        st.markdown(color_message_two, unsafe_allow_html=True)
        st.markdown(color_message_three, unsafe_allow_html=True)
        st.write('')
        st.write('')


        # Display the labels again, they'll be needed for the explanation.
        st.write('_Labels for each tweet given by clustering algorithm:_')
        st.write(f'{list_of_labels}')
        st.write('')
        st.write('')

        # In depth descriptions.
        st.write(self.m_post_plot_description_two)
        st.write('')
        st.write(self.m_post_plot_description_three)
    
        ''' Get the label separated tweets as groups with their corresponding label names. The label names were found out when the Plot function
            was called earlier. See that function for more details in the comments regarding how the text labels/sentiment was returned.
        
            To display this information with ease, create dataframes of each to display to the user. For example, the neutral portion of tweets
            that the clustering algorithm BELIEVED were neutral (after it used the fit function), will be displayed here. As well as the other
            groups positive and negative. '''
        st.write('')
        st.write('')
        st.write('')
        label_separated_tweets = self.GetLabelSeparatedTweets()
        label_separated_data_frames = []

        # Create separate dataframes of each group of tweets that will be used for easy display below.
        for entry in label_separated_tweets:
            label_separated_data_frames.append(pd.DataFrame(data=entry[1], columns=[entry[0]]))


        for i in range(len(label_separated_data_frames)):
            # Get the dataframe via index first.
            df = label_separated_data_frames[i]

            ''' Get the column name so this will be displayed ABOVE the dataframe itself. But later it'll be displayed in color. It's a SMALL
                extra addition but it will visually help the user see which tweets the clustering algorithm labeled positive, neutral and negative.
                A list is retrieved but these dataframes were created with only ONE column. So later accessing the only column by index 0 is fine. '''
            column_name = df.columns.tolist()

            ''' Color it here with the helper function. This function was used before and displayed colored text right after the plot earlier.
                Pass it the color and the column name via index, as stated in the previous comment. '''
            st.markdown(self.GetColoredText(self.m_colors[i][1], column_name[0]), unsafe_allow_html=True)

            # Now display the whole dataframe under it.
            st.write(df)
    



    def GetColoredText(self, color_to_use, text):
        return f'<p style="color:{color_to_use};">{text}</p>'



    ''' Return the tweets that were retrieved from the given dataframe in the above function. They were retrieved via a number indicating their
        sentiment. 0, 1, or 2. Like so: "self.m_zero_tweets = df[df[labels_column_name] == 0][tweets_column_name].tolist()" Also it would be 
        wise to return what associated label goes WITH them. This can be done by using the colors they are assigned. The colors list will provide 
        tuples of both color strings and those colors hex values but at the moment what's needed is the colors string value.
        
        So with everything said, tweets with a certain label (0, 1, or 2) will be returned with their assigned string colors. '''
    def GetLabelSeparatedTweets(self):
        first_set = (self.m_colors[0][0], self.m_zero_tweets)
        second_set = (self.m_colors[1][0], self.m_one_tweets)
        third_set = (self.m_colors[2][0], self.m_two_tweets)

        return first_set, second_set, third_set



    ''' This function will be used because it will decide what group gets what color based on their sentiment. The original sets of positive, neutral,
        and negative lists of tweets were saved in the "Clustering" step. That was important because those tweets were the starting point to even
        create a plot, and therefore create this entire project. They'll be used here again because for building the plot that ONLY has the example tweets
        (the original set), the chosen clustering algorithm (such as k means), will make predictions/labels for each tweet. So each tweet has a value.
        
        Let's say we gather ALL the tweets that the clustering algorithm has labeled as 0. How well do THOSE tweets match the positive, neutral, or 
        negative tweets? THAT'S the question that this function answers. Give it a set of tweets that were organized by the predictions/labels and let
        it figure out whether it matches the ORIGINAL positive, neutral, or negative sentiment tweets and return a color based on that. '''
    def GetLabelForTweets(self, tweets_to_test):
        # First get all the saved groups of tweets in session state.
        pos_tweets = st.session_state.m_positive_tweets 
        neu_tweets = st.session_state.m_neutral_tweets 
        neg_tweets = st.session_state.m_negative_tweets 

        # A counter for detecting matches.
        pos_counter = 0
        neu_counter = 0
        neg_counter = 0

        # Comparison loop
        for i in range(len(pos_tweets)):
            for x in range(len(tweets_to_test)):
                if pos_tweets[i] == tweets_to_test[x]:
                    pos_counter += 1
        
        for i in range(len(neu_tweets)):
            for x in range(len(tweets_to_test)):
                if neu_tweets[i] == tweets_to_test[x]:
                    neu_counter += 1

        for i in range(len(neg_tweets)):
            for x in range(len(tweets_to_test)):
                if neg_tweets[i] == tweets_to_test[x]:
                    neg_counter += 1   
        
        ''' Based on what value is greater, a color will be returned. If positive, return green. If neutral, return
            yellow. If negative, return red. Stop light colors. '''
        selected_color = None

        if (pos_counter > neu_counter) and (pos_counter > neg_counter):
            selected_color = ('Positive', '#00FF00')
        
        elif (neu_counter > pos_counter) and (neu_counter > neg_counter):
            selected_color = ('Neutral', '#FFFF00')
        
        elif (neg_counter > pos_counter) and (neg_counter > neu_counter):
            selected_color = ('Negative', '#FF0000')

        print(f'---Color selected: {selected_color[0]}---\n')

        return selected_color