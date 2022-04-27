import abc
import streamlit as st
import random

from models_strategy_pattern.model_strategy import ModelStrategy
from models_strategy_pattern.model_types import ModelTypes

'''
            Data 

This is a base class that will be used to implement functions that all derived classes will override. 
Since all classes will need to perform certain operations, this makes sense to do. In main.py there
should definitely be one "set of functions" that the main.py code can rely on when dealing with
any class. The interface below is extremely simple.

    Functions:

        1) Constructor - Initialize class data.

        2) Display - Used for streamlit, this function will call functions such as st.write() and more.
            Streamlit is mainly for show and this ensures things will be shown.

    
    Variables:

        1) m_details - Every class in the pipeline will at LEAST have to give one description on itself.
            More is definitely acceptable but universally all derived classes will utilize this to explain
            to the user what the class does.


    
'''

class Data:
    def __init__(self):
        self.m_details = ''' '''

        # self.m_key_numbers_as_str = []
        # self.m_range_start = 0
        # self.m_range_end = 1000

        # self.m_display_page_info = True


    def Display(self):
        pass


    def ModelCompatibilityCheck(self):
        return True

    
    def IsLastClass(self):
        return False


    ''' For every subclass of Data, since each subclass is participating in the chain of responsibility pattern, get the model
        type itself to check if the model will need to bother with a particular step. The subclass will handle the rest like
        going to the next class. This function keeps the job simple, it gets the type and lets calling class know if the model
        belongs here or not. '''
    def ModelTypeCheck(self, model_class, calling_class_name, type_allowed_for_this_class, show_steps=False):
        if show_steps is True:
            print('\n\n---Model type check---')

        model_type = model_class.GetType()

        if show_steps is True:
            print(f'{model_class} type is {model_type}')

        if model_type != type_allowed_for_this_class:
            # Print quick message about unnecessary class for this step and go to next class.
            if show_steps is True:
                print(f"{calling_class_name} doesn't use type {model_type}. Correct type for {calling_class_name} is {type_allowed_for_this_class}. Going to next class!")

            return False

        if show_steps is True:
            print(f"{calling_class_name} does use type {model_type}. Continuing.")

        if show_steps is True:
            print('---End model type check---\n\n')


        return True


    
    # def CheckCompatibility(self, model, type_allowed_for_this_class):
    #     if model is None:
    #         return True
        
    #     model_type = model.GetType()

    #     if model_type == type_allowed_for_this_class:
    #         return True
        

    #     return False