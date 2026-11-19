"""MobiHet with a private benign-vs-timeDelay pairwise auxiliary objective."""
from methods.mobihet import Method as MobiHet


class Method(MobiHet):
    name = "mobihet_pair"
    pairwise_aux = True
