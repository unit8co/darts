"""
Dropout for Torch Models
-----------------


"""

from abc import ABC

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class Dropout(ABC):
    def __init__(self, dropout_function):
        """
        Abstract class for a dropout model layer.
        dropout_function must exist in every instance 
        Dropout Class.
        
        """
        self.dropout_function = dropout_function
        


        
class TorchNativeDropout(Dropout):
    """Torch Native (nn.Dropout) Dropout layer to be used with 
    different torch models. 
    """
    def __init__(self, dropout: float=0.2):
        self.dropout_function = nn.Dropout(dropout)
        super().__init__(self.dropout_function)
        

class McCompatibleDropout(Dropout):
    """Monte Carlo Dropout layer to be used with 
    different torch models. 
    
    """
    def __init__(self, dropout: float=0.2):
        self.dropout_function = McCompatibleDropoutModule(dropout)
        super().__init__(self.dropout_function)
        
        
    
    

class McCompatibleDropoutModule(nn.Dropout):
    """Defines Monte Carlo dropout Module as defined
    in the paper https://arxiv.org/pdf/1506.02142.pdf. 
    In summary, This techniques uses of the regular dropout 
    which can be interpreted as a Bayesian approximation of 
    a well-known probabilistic model: the Gaussian process. 
    We can treat the many different networks 
    (with different neurons dropped out) as Monte Carlo samples 
    from the space of all available models. This provides mathematical 
    grounds to reason about the model’s uncertainty and, as it turns out, 
    often improves its performance.
    
    """
    mc_dropout_enabled = True
    
    def set_dropout_status(self, mc_dropout_enabled: bool):
        self.mc_dropout_enabled = mc_dropout_enabled
    
    def train(self, mode:bool=True):
        if(mode):  # in train mode, keep dropout as is
            self.mc_dropout_enabled = True
        if(not mode): # in eval mode, bank on the mc_dropout_enabled flag
            pass
        
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.mc_dropout_enabled, self.inplace)

