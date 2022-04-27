import streamlit as st
from sklearn.cluster import KMeans
from clustering_strategy_pattern.clustering_strategy import ClusteringStrategy


class SKL_KMeans(ClusteringStrategy):
    def __init__(self, random_state):
        super().__init__('K Means', random_state)

        # Initialize model itself.
        self.m_clustering_algorithm = KMeans(n_clusters=self.m_num_of_clusters, random_state=self.m_random_state)

    
    def Fit(self, embeddings):
        # Display to user the clustering algorithm will begin.
        st.write(f'##### Apply {self.m_name} clustering algorithm.')

        # Description gives more details.
        st.write(self.m_clustering_description)
        st.write('')

        # Fit on the embeddings (like umap) found when model called the function "GetUmapReducedDimension".
        self.m_clustering_algorithm.fit(embeddings)

        st.write(f'Labels for each tweet from clustering algorithm: {self.m_clustering_algorithm.labels_}')

    
    def GetLabels(self):
        return self.m_clustering_algorithm.labels_


    def GetLabelsAsList(self):
        return self.m_clustering_algorithm.labels_.tolist()