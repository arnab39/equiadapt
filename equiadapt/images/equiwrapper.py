
from equiadapt.common.basewrapper import BaseWrapper

class ImageEquiWrapper(BaseWrapper):
    def __init__(self, canonicalization_network, **kwargs):
        super().__init__(canonicalization_network, **kwargs)
        
    def canonize(self, x, **kwargs):
        self.canonicalization_network.canonicalize(x, **kwargs)
        