from abc import ABC, abstractmethod


'''
        Clustering Strategy

This is the base class to all derived scikit learn, and POTENTIALLY other libraries clustering algorithms, such as k means. Mostly all 
clustering algorithms that give the ability to SPECIFY number of clusters will be included here to absolutely ENSURE that they follow 
only 3 clusters for positive, neutral, and negative. The actual interface will be extremely simple but this is a good way to keep the 
code as neat as possible, making sure any and all clustering algorithms are in a subclass.


'''

class ClusteringStrategy(ABC):
    def __init__(self, name, random_state) -> None:
        self.m_name = name

        # Random state can be applied to clustering algorithms like k_means.
        self.m_random_state = random_state

        self.m_num_of_clusters = 3

        self.m_clustering_description = ''' Since this is an unsupervised learning problem, clustering algorithms will be
        applied so the data can be organized in some fashion. 3 clusters will be created, positive, neutral, and negative.
        Labels will be provided and those numbers will indicate which of the 3 clusters a particular tweet belongs too. '''

        self.m_clustering_algorithm = None




    ''' 

            All functions below are for derived classes to override.

            1) Fit - Compute the algorithm.

            2) GetLabels - Updates dataframe with new labels, referenced in St_AllMpnetBaseV2.py.

            3) GetLabelsAsList - Displays new labels in similar fashion, also referenced in St_AllMpnetBaseV2.py.
    
    '''
    @abstractmethod
    def Fit(self, embeddings):
        pass


    @abstractmethod
    def GetLabels(self):
        pass


    @abstractmethod
    def GetLabelsAsList(self):
        pass


    
    ''' 

                All functions below are implemented in this class. All dervived objects need not override.
    
    '''
    def GetName(self):
        return self.m_name