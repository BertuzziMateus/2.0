import backend_select
import numpy as np
import pandas as pd

# Selected backend (can be numpy or cupy)
xp = backend_select.get_array_module() 

class Fluid:
    def __init__(self,
                 pressures: list,
                 bo: list,
                 u_o: list,
                 pb: float,
                 **kwargs
                 ):

        p_raw = np.array(pressures, dtype=float)
        bo_raw = np.array(bo, dtype=float)
        uo_raw = np.array(u_o, dtype=float)

        sorted_indices = np.argsort(p_raw)

        # 2. Store in backend arrays (xp) already sorted
        self.table_p = xp.array(p_raw[sorted_indices])
        self.table_bo = xp.array(bo_raw[sorted_indices])
        self.table_uo = xp.array(uo_raw[sorted_indices])
        
        self.pb = float(pb) # Bubble point is a scalar

        # Consistency check
        if not (self.table_p.size == self.table_bo.size == self.table_uo.size):
            raise ValueError("Pressure, Bo, and Viscosity arrays must have the same size.")

    def get_pvt_properties(self, pressure_field):
        """
        Interpolates Bo and Viscosity for a given pressure field.
        """

        p_field = xp.asarray(pressure_field)
        
        # Now we pass p_field (which is an array) to interp
        bo_vals = xp.interp(p_field, self.table_p, self.table_bo)
        uo_vals = xp.interp(p_field, self.table_p, self.table_uo)
        
        return bo_vals, uo_vals

    @classmethod
    def from_file(cls, filepath: str, bubble_point: float, sep=';', decimal=','):
        """
        Loads PVT properties from a CSV.
        Expects columns containing 'PRESSURE', 'Bo' or 'FACTOR', and 'VISCOSITY'.
        """
        try:
            df = cls._read_csv_content(filepath, sep, decimal)
            print(f'-> Loading fluid from: {filepath}')
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            raise

        df.columns = [c.upper().strip() for c in df.columns]

        # Intelligent column identification (looks for substrings)
        try:
            col_p = next(c for c in df.columns if 'PRESS' in c)
            # Tries to find Bo column (Formation Volume Factor)
            col_bo = next(c for c in df.columns if 'BO' in c or 'FACTOR' in c or 'FATOR' in c)
            # Tries to find Viscosity column
            col_uo = next(c for c in df.columns if 'VISC' in c or 'U_O' in c or 'MI' in c)
            
            return cls(
                pressures=df[col_p].values,
                bo=df[col_bo].values,
                u_o=df[col_uo].values,
                pb=bubble_point
            )

        except StopIteration:
            raise ValueError(f"Could not identify required columns (Pressure, Bo, Viscosity) in file {filepath}. Found columns: {df.columns}")

    @staticmethod
    def _read_csv_content(filepath, sep, decimal):
        # Reading always via Pandas (CPU)
        return pd.read_csv(filepath, sep=sep, decimal=decimal)