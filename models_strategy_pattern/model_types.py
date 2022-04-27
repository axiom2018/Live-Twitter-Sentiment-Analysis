from enum import Enum

''' To help with the chain of responsibility pattern know exactly when to pass the request onto the next in line, these types 
    will be specified. A simple library model type with say a Vader model won't be dealing with embeddings and therefore will 
    not need to take those steps regarding embedding. '''

class ModelTypes(Enum):
    SimpleLibrary=0
    Embedding=1