import backend_select
import numpy as np
import pandas as pd

# Backend selecionado (pode ser numpy ou cupy)
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

        # 2. Armazenar nos arrays do backend (xp) já ordenados
        self.table_p = xp.array(p_raw[sorted_indices])
        self.table_bo = xp.array(bo_raw[sorted_indices])
        self.table_uo = xp.array(uo_raw[sorted_indices])
        
        self.pb = float(pb) # Bubble point é um escalar

        # Verificação de consistência
        if not (self.table_p.size == self.table_bo.size == self.table_uo.size):
            raise ValueError("Os arrays de Pressão, Bo e Viscosidade devem ter o mesmo tamanho.")

    def get_pvt_properties(self, pressure_field):
        """
        Interpola Bo e Viscosidade para um dado campo de pressão.
        """

        p_field = xp.asarray(pressure_field)
        
        # Agora passamos p_field (que é um array) para o interp
        bo_vals = xp.interp(p_field, self.table_p, self.table_bo)
        uo_vals = xp.interp(p_field, self.table_p, self.table_uo)
        
        return bo_vals, uo_vals

    @classmethod
    def from_file(cls, filepath: str,bubble_point:float, sep=';', decimal=','):
        """
        Carrega propriedades PVT de um CSV.
        Espera colunas contendo 'PRESSURE', 'Bo' ou 'FACTOR', e 'VISCOSITY'.
        """
        try:
            df = cls._read_csv_content(filepath, sep, decimal)
            print(f'-> Carregando fluido de: {filepath}')
        except FileNotFoundError:
            print(f"Erro: Arquivo '{filepath}' não encontrado.")
            raise

        df.columns = [c.upper().strip() for c in df.columns]

        # Identificação inteligente das colunas (procura substrings)
        try:
            col_p = next(c for c in df.columns if 'PRESS' in c)
            # Tenta encontrar coluna de Bo (Fator volume formação)
            col_bo = next(c for c in df.columns if 'BO' in c or 'FACTOR' in c or 'FATOR' in c)
            # Tenta encontrar coluna de Viscosidade
            col_uo = next(c for c in df.columns if 'VISC' in c or 'U_O' in c or 'MI' in c)
            
    

            return cls(
                pressures=df[col_p].values,
                bo=df[col_bo].values,
                u_o=df[col_uo].values,
                pb=bubble_point
            )

        except StopIteration:
            raise ValueError(f"Não foi possível identificar as colunas necessárias (Pressure, Bo, Viscosity) no arquivo {filepath}. Colunas encontradas: {df.columns}")

    @staticmethod
    def _read_csv_content(filepath, sep, decimal):
        # Leitura sempre via Pandas (CPU)
        return pd.read_csv(filepath, sep=sep, decimal=decimal)