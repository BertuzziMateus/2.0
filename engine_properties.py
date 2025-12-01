import backend_select
import re
import numpy as np

xp = backend_select.get_array_module()


class ReservoirProperties:
    
    def __init__(self,
                 porosity: list,
                 ntg: list,
                 permx: list,
                 permy: list,
                 permz: list,
                 **kwargs):
        """
        Inicializa as propriedades do reservatório.
        Converte listas ou np.arrays para o backend selecionado (xp).
        """

        # Porosidade
        self.porosity = xp.array(porosity, dtype=float).flatten()
        self.n_cells = self.porosity.size

        # NTG
        if ntg is not None:
            self.ntg = xp.array(ntg, dtype=float).flatten()
        else:
            self.ntg = xp.ones_like(self.porosity)

        # Permeabilidades
        self.permx = xp.array(permx, dtype=float).flatten() if permx is not None else None
        self.permy = xp.array(permy, dtype=float).flatten() if permy is not None else None
        self.permz = xp.array(permz, dtype=float).flatten() if permz is not None else None


    # -------------------------------------------------------------

    def validate_with_grid(self, grid):
        """
        Valida se o número de células das propriedades corresponde ao grid.
        """

        print(f"Validando propriedades: Grid ({grid.nt}) vs Props ({self.n_cells})")

        if self.n_cells != grid.nt:
            raise ValueError(
                f"[ERRO] Dimensões incompatíveis!\n"
                f"Grid: {grid.nt} ({grid.nx} x {grid.ny} x {grid.nz})\n"
                f"Propriedades: {self.n_cells}"
            )

        if self.ntg.size != grid.nt:
             raise ValueError(f"[ERRO] NTG ({self.ntg.size}) difere do grid.")

        if self.permx is not None and self.permx.size != grid.nt:
            raise ValueError(f"[ERRO] PERMX ({self.permx.size}) difere do grid.")

        print(">> Validação concluída: OK.\n")
        return True

    # -------------------------------------------------------------

    @classmethod
    def from_file(cls, filepath: str):
        """
        Lê um arquivo GRDECL e retorna as propriedades convertidas para o backend.
        """

        print(f"Lendo arquivo GRDECL: {filepath}")

        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            raise ValueError(f"[ERRO] Arquivo {filepath} não encontrado.")

        data = cls._parse_grdecl_content(content)

        # Coleta segura das propriedades
        porosity = data.get('PORO')
        ntg      = data.get('NTG')
        permx    = data.get('PERMX')
        permy    = data.get('PERMY')
        permz    = data.get('PERMZ')

        if porosity is None:
            raise ValueError("[ERRO] Arquivo não contém PORO.")

        return cls(
            porosity=porosity,
            ntg=ntg,
            permx=permx,
            permy=permy,
            permz=permz
        )

    # -------------------------------------------------------------

    @staticmethod
    def _parse_grdecl_content(content):
        """
        Parser rápido e seguro para arquivos GRDECL.
        Suporta expressões tipo: 100*0.25
        """

        target_keywords = ['PORO', 'NTG', 'PERMX', 'PERMY', 'PERMZ', 'ACTNUM', 'ROCK']
        extracted_data = {}

        # Remove comentários "-- ..."
        content_clean = re.sub(r'--.*$', '', content, flags=re.MULTILINE)

        # Tokeniza por espaços ou "/"
        tokens = re.split(r'\s+|/', content_clean)

        current_key = None
        buffer_vals = []

        for token in tokens:
            if not token:
                continue

            token_up = token.upper()

            # Se encontrar keyword nova → salva a anterior
            if token_up in target_keywords:
                if current_key and buffer_vals:
                    extracted_data[current_key] = np.array(buffer_vals, dtype=float)

                current_key = token_up
                buffer_vals = []
                continue

            # Coleta valores enquanto uma keyword está ativa
            if current_key:

                # Caso "100*0.25"
                if '*' in token:
                    try:
                        count_str, val_str = token.split('*')
                        count = int(count_str)
                        val = float(val_str)
                        buffer_vals.extend([val] * count)
                    except:
                        # token inválido, ignora
                        pass

                else:
                    try:
                        buffer_vals.append(float(token))
                    except:
                        pass

        # Armazena última keyword lida
        if current_key and buffer_vals:
            extracted_data[current_key] = np.array(buffer_vals, dtype=float)

        return extracted_data
