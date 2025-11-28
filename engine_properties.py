
import backend_select
import re
import numpy as np 

xp = backend_select.get_array_module() 


class ReservoirProperties:
    
    def __init__(self,
                 porosity:list,
                 ntg:list,
                 permx:list,
                 permy:list,
                 permz:list,
                 **kwargs
                 ):
        """
        Inicializa as propriedades. 
        Recebe dados (seja lista ou numpy array) e converte para o BACKEND (xp).
        """
        # Porosidade
        self.porosity = xp.array(porosity, dtype=float).flatten()
        self.n_cells = self.porosity.size 
        
        # NTG (Net-to-Gross)
        if ntg is not None:
            self.ntg = xp.array(ntg, dtype=float).flatten()
        else:
            self.ntg = xp.ones_like(self.porosity)

        # Permeabilidades
        self.permx = xp.array(permx, dtype=float).flatten() if permx is not None else None
        self.permy = xp.array(permy, dtype=float).flatten() if permy is not None else None
        self.permz = xp.array(permz, dtype=float).flatten() if permz is not None else None

    def validate_with_grid(self, grid):
        """
        Verifica consistência dimensional entre Propriedades e Grid.
        """
        print(f"Validando: Grid ({grid.nt}) vs Props ({self.n_cells})...")
        
        if self.n_cells != grid.nt:
            raise ValueError(
                f"[ERRO] Dimensões incompatíveis!\n"
                f"Grid: {grid.nt} ({grid.nx}x{grid.ny}x{grid.nz})\n"
                f"Props (PORO): {self.n_cells}"
            )
            
        if self.ntg.size != grid.nt:
             raise ValueError(f"NTG ({self.ntg.size}) difere do Grid.")
             
        if self.permx is not None and self.permx.size != grid.nt:
            raise ValueError(f"PERMX ({self.permx.size}) difere do Grid.")
            
        print(">> Validação: OK.")
        return True

    @classmethod
    def from_file(cls, filepath: str):
        """
        Lê arquivo GRDECL e retorna instância da classe.
        """
        print(f"Lendo arquivo: {filepath}")
        with open(filepath, 'r') as f:
            content = f.read()
            
        # O parser retorna dicionário com arrays
        data = cls._parse_grdecl_content(content)

        try:
            porosity=data.get('PORO')
            ntg=data.get('NTG')
            permx=data.get('PERMX')
            permy=data.get('PERMY')
            permz=data.get('PERMZ')
        except:
            raise ValueError(f"Erro to get properties.")
        


        # O __init__ converterá esses arrays Numpy para arrays XP (Backend)
        return cls(
            porosity=porosity,
            ntg=ntg,
            permx=permx,
            permy=permy,
            permz=permz
        )

    @staticmethod
    def _parse_grdecl_content(content):
        """
        Parser otimizado: Usa sempre NumPy (CPU) para processamento de texto.
        """
        target_keywords = ['PORO', 'NTG', 'PERMX', 'PERMY', 'PERMZ', 'ACTNUM']
        extracted_data = {}
        
        content_clean = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
        tokens = re.split(r'\s+|/', content_clean)
        
        current_key = None
        buffer_vals = []
        
        for token in tokens:
            if not token: continue 
            
            token_upper = token.upper()
            
            if token_upper in target_keywords:
                if current_key and buffer_vals:
                    extracted_data[current_key] = np.array(buffer_vals, dtype=float)
                
                current_key = token_upper
                buffer_vals = []
                continue
            
            if current_key:
                if '*' in token:
                    try:
                        count_str, val_str = token.split('*')
                        count = int(count_str)
                        val = float(val_str)

                        buffer_vals.extend([val] * count)
                    except ValueError:
                        pass 
                else:
                    try:
                        val = float(token)
                        buffer_vals.append(val)
                    except ValueError:
                        pass 

        if current_key and buffer_vals:
            extracted_data[current_key] = np.array(buffer_vals, dtype=float)
            
        return extracted_data