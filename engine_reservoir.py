import backend_select
import re
import numpy as np

# Backend selecionado (pode ser numpy ou cupy)
xp = backend_select.get_array_module() 


class Field:

    def __init__(self,
                 grid,
                 properties,
                 fluid,
                 ct,               
                 ):
        
        self.grid = grid
        self.fluid = fluid
        self.properties = properties
        self.ct = ct
        
        pass


