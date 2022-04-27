import streamlit as st
import pandas as pd
import numpy as np
import math


from Etc.ScatterPlot import ScatterPlot

'''
            Mean Classification

This class will take a slightly different route regarding labeling new Tweets pulled straight from Twitter. The project starts
with example tweets that I've manually labeled and in a cluster they look great. But for the new tweets, they may or may not
be mislabeled. With that said, this will be an attempt to take a new tweets 2d points and compare it to the mean of every
other group. The groups are positive, neutral and negative. Whichever distance is lower, being the distance from the new tweet
2d point and the mean of all other groups, that's the new class the new tweet will be assigned.


'''

class MeanClassification:
    def __init__(self) -> None:
        self.m_scatter_plot = ScatterPlot()

    def ClassifyNewTweets(self, df, num_for_unlabeled_tweets):
        st.write('### Classify tweets with mean.')
        st.write('')
        st.write('')

        ''' ---Getting the mean---
        
            Zero x/y coordinates:

            1) 8.021038055419922, 7.957052230834961, 8.278220176696777, 8.566353797912598, 7.761924743652344, 
                8.667365074157715, 8.777849197387695, 8.3773193359375, 8.151694297790527

            2) 13.4159574508667, 12.083443641662598, 13.048285484313965, 13.179594039916992, 12.498967170715332, 
                11.857135772705078, 12.379222869873047, 12.22765827178955, 12.564867973327637

            The FUNCTION says the mean is: [[ 8.28431299 12.58390363]]

            Website results:

            1) https://www.calculatorsoup.com/calculators/statistics/mean-median-mode.php

                Gives 8.2843129899767, 12.583903630575

            2) https://www.calculator.net/mean-median-mode-range-calculator.html

                Gives 8.2843129899767, 12.583903630575

            So the function WORKS! I should now be able to 
        
        '''

        # -----Get the points from each label of tweets.
        zero_x_points = df[df['labels'] == 0]['x'].tolist()
        zero_y_points = df[df['labels'] == 0]['y'].tolist()

        one_x_points = df[df['labels'] == 1]['x'].tolist()
        one_y_points = df[df['labels'] == 1]['y'].tolist()

        two_x_points = df[df['labels'] == 2]['x'].tolist()
        two_y_points = df[df['labels'] == 2]['y'].tolist()

        ''' Pass each set (0's, 1's and 2's) of points x and y coordinates to function to find mean. In "[[ ]]" format. 
            Pass [0] to get the inner list, and of course [0] and [1] WITH that to access specific values. '''
        zero_mean = self.GetMean(zero_x_points, zero_y_points)
        one_mean = self.GetMean(one_x_points, one_y_points)
        two_mean = self.GetMean(two_x_points, two_y_points)

        st.write(f'##### Mean data: ')
        st.write(f'Mean for zero x/y points: {zero_mean[0]}.')
        st.write(f'Mean for one x/y points: {one_mean[0]}.')
        st.write(f'Mean for two x/y points: {two_mean[0]}.')
        st.write('')
        st.write('')
 

        ''' ---Getting the distance---

            The zero mean was: [[ 8.28431299 12.58390363]], so let's find the distance between IT and a dummy point.
            The dummy point is defined below.

            The CODE BELOW says the distance is: 8.264520923439619

            Website results:

            1) https://www.calculator.net/distance-calculator.html

                Gives 8.2645209229217



            Find the shortest distance between -1 labeled rows in the dataframe. As of 4/17/2022, -1 is the chosen
            value to find rows of "new tweets" that were pulled straight from twitter and haven't yet received a 
            sentiment.
        
        '''

        ''' The calling class created a dataframe where new tweets had a certain label to tell them apart from others. 
            It's provided as an argument here so this class can RETRIEVE all new tweets, AND the example tweets in
            one dataframe. 
            
            Reset the index because looping over it will be necessary and starting from the 0 index is more sensible. '''
        result = df.loc[df['labels'] == num_for_unlabeled_tweets]
        result.reset_index(inplace=True)
        result.drop(columns=["index"], inplace=True)
        st.write('##### Dataframe of unlabeled new tweets: ')
        st.write(result)
        st.write(type(result))

        # Will be necessary for GetShortestDistance.
        list_of_means = [zero_mean, one_mean, two_mean]

        for i in range(len(result)):
            # Get every x and y point available in the unlabeled tweet section in this specific format.
            point = [result['x'][i], result['y'][i]]
            # st.write(f'Point: {point}')

            # Get the shortest distance list of values between the above point and the means for positive, neutral, and negative.
            shortest_dist_list = self.GetShortestDistance(list_of_means, point)
            # st.write(f'Shortest distance list: {shortest_dist_list}')

            # Now the shortest distance itself
            min_dist = min(shortest_dist_list)
            # st.write(f'min_dist: {min_dist}')

            ''' The shortest distance list and the list made of means (created before this for loop) are useful when brought together.
                If the list of means has, in order: zero mean, one mean, and two mean, then when they get passed to the function to
                get the shortest distance it'll bring back a list with distances regarding every previously mentioned mean in that
                order. So if the shortest distance between all the means and say point [5, 5] is at index 0 in the list that the
                function GetShortestDistance returns, that index correlates to being the zero mean, so this point should be assigned
                a 0. '''
            min_dist_index = shortest_dist_list.index(min_dist)
            # st.write(f'min_dist_index: {min_dist_index}')

            # st.write(result[i])
            result['labels'][i] = min_dist_index


        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('##### Dataframe of labeled new tweets: ')
        st.write(result)


        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('##### Drop the unlabeled tweets from main dataframe: ')
        df = df[df['labels'] != num_for_unlabeled_tweets]
        st.write(df)


        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('##### Combine dataframe of example tweets with dataframe of new tweets: ')
        frames = [df, result]
        combined_df = pd.concat(frames)
        combined_df.reset_index(inplace=True)
        combined_df.drop(columns=["index"], inplace=True)
        st.write(combined_df)

        # Plot
        self.m_scatter_plot.Plot(combined_df, 'labels', 'tweets', (9, 9))







        # -----Get the distance between the following example point and one of the 3 main points.
        # example_point = [5, 5]
        # dist = math.dist(zero_mean[0], example_point)
        # st.write(f'Distance between {zero_mean} and {example_point} is {dist}.')

        # # -----Get the distance between the example point and 3 main points.
        # list_of_means = [zero_mean, one_mean, two_mean]
        # shortest_dist = self.GetShortestDistance(list_of_means, example_point, True)
        # st.write(f'Shortest distance is: {shortest_dist}')





    def GetMean(self, x_value_list, y_value_list, print_details_in_console=False):
        if print_details_in_console is True:
            print(f'(GetMean function) X values: {x_value_list}.\n')
            print(f'(GetMean function) Y values: {y_value_list}.\n')

        list_of_paired_values = []

        for i in range(len(x_value_list)):
            list_of_paired_values.append([x_value_list[i], y_value_list[i]])

        num_of_values_to_display = 5
        if print_details_in_console is True:
            print(f'(GetMean function) Displaying {num_of_values_to_display} values of list: {list_of_paired_values[:num_of_values_to_display]}.\n')
            print(f'(GetMean function) Index 0: {list_of_paired_values[0]}.\n')

        a = np.array([list_of_paired_values])
        mean_value = a.mean(axis=1)

        if print_details_in_console is True:
            print(f'(GetMean function) Mean value is {mean_value}.\n')

        return mean_value



    def GetShortestDistance(self, mean_points, cur_point, show_steps=False):
        # Get a list of the distances in order to figure which one is shortest.
        list_of_distances = []

        # mp = [m]ean [p]oint.
        for mp in mean_points:
            # Calculate distance, then later add to list.
            dist = math.dist(mp[0], cur_point)

            if show_steps is True:
                st.write(f'(GetShortestDistance function) Distance between {mp} and {cur_point} is {dist}.')
            
            list_of_distances.append(dist)

        # Easy way to get the minimum with min.
        # return min(list_of_distances)
        return list_of_distances