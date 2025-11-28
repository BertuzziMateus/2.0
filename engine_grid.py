import backend_select
import re
import numpy as np

# Backend selecionado (pode ser numpy ou cupy)
xp = backend_select.get_array_module() 

class StructuredGrid:
    def __init__(self,
                 x_length: float,  
                 y_length: float, 
                 thickness: list, 
                 nx: int, 
                 ny: int, 
                 nz: int,
                 actnum: list,                 
                 ):
        
        self.x_length = x_length
        self.y_length = y_length
        

        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        if len(thickness) != nz:
            raise ValueError(f'Length of thickness ({len(thickness)}) must equal Nz ({nz}).')
        
        self.nt = nx * ny * nz
        
        # --- Geometria ---
        self.dx = x_length / nx
        self.dy = y_length / ny

        # Espessuras
        self.thick = xp.array(thickness, dtype=float)
        
        # Expande a espessura para todas as células (1D Flattened)
        self.dz = xp.repeat(self.thick, nx*ny)
        
        # Volume
        self.vb = self.dx * self.dy * self.dz

        # Áreas das faces (para transmissibilidade)
        self.ax = self.dy * self.dz
        self.ay = self.dx * self.dz
        # Az constante para grid cartesiano regular
        self.az = xp.full((self.nt,), self.dx * self.dy, dtype=float)

        self.actnum = actnum

        if len(actnum) != self.nt:
            raise ValueError(f"Lenght of actnum array must be equal the total cell in the grid: Nt: {self.nt}")


    @classmethod
    def from_file(cls, filepath: str):
        with open(filepath, 'r') as f:
            content = f.read()

        data = cls._parcel_grid_content(content)


        dimens = np.fromstring(data.get('DIMENS', '1 1 1'), sep=' ', dtype=int)
        nx, ny, nz = dimens[0], dimens[1], dimens[2]

        coordx = np.fromstring(data.get('COORDX', '0 1'), sep=' ', dtype=float)
        coordy = np.fromstring(data.get('COORDY', '0 1'), sep=' ', dtype=float)
        x_len = coordx[1] - coordx[0]
        y_len = coordy[1] - coordy[0]

        thick_data = np.fromstring(data.get('THICKNESS', '1.0'), sep=' ', dtype=float)


        actnum_data = np.fromstring(data.get('ACTNUM', ''), sep=' ', dtype=int)

        return cls(
            x_length=x_len,
            y_length=y_len,
            thickness=thick_data,
            nx=nx, ny=ny, nz=nz,
            actnum=actnum_data
        )

    @staticmethod
    def _parcel_grid_content(content):
        data = {}
        # Regex captura PalavraChave + conteúdo até a barra /
        pattern = re.compile(r'(\w+)\s+([\s\S]*?)(?:/)')
        matches = pattern.findall(content)
        for key, val in matches:
            data[key] = val.replace('\n', ' ').strip()
        return data
        