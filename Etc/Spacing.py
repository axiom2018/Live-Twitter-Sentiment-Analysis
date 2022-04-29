import streamlit as st


''' 
            Spacing

This will provide one utility function in this class and make it available to all derived classes. This solves the problem 
of having the same portion of code copied and pasted in several groups of files.

'''

class Spacing:
    def __init__(self) -> None:
        pass

    # Help give spaces between portions of text and such on screen.
    def ApplySpacingOnScreen(self, amount_of_spaces):
        for i in range(amount_of_spaces):
            st.write('')