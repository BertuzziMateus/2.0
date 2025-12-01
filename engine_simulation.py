import backend_select
xp = backend_select.get_array_module()

# Se backend for Cupy, importe as versões GPU
if xp.__name__ == "cupy":
    from cupyx.scipy.sparse import csr_matrix
    from cupyx.scipy.sparse.linalg import cg
else:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import cg

# ------------------------------
# Conversão de unidades → SI
# ------------------------------
kgf_cm2_to_Pa = 98066.5
md_to_m2 = 9.869233e-16
cp_to_Pa_s = 0.001
day_to_sec = 86400


class SimulationEngine:
    def __init__(self, field, wells, simulation_time, time_step):

        self.field = field
        self.wells = wells
        
        self.simulation_time = simulation_time * day_to_sec     # dias → s
        self.time_step = time_step * day_to_sec


    # ---------------------------------------------------------
    # PVT → retorna bo, uo no backend
    # ---------------------------------------------------------
    def get_fluid_proprerties(self, P_old):

        P_SI = xp.array(P_old)      # já está em Pa
        return self.field.fluid.get_pvt_properties(P_SI)
    

    # ---------------------------------------------------------
    # Gamma (acúmulo)
    # ---------------------------------------------------------
    def get_gamma_term(self, P_old):

        bo, uo = self.get_fluid_proprerties(P_old/ kgf_cm2_to_Pa)

        Vb = self.field.grid.vb
        poro = self.field.properties.porosity
        ct = self.field.ct / kgf_cm2_to_Pa   # converte para 1/Pa
        dt = self.time_step * day_to_sec

        gamma = (Vb * poro * ct) / (bo * dt)
        return gamma


    # ---------------------------------------------------------
    # Monta heptadiagonal (retorna COO)
    # ---------------------------------------------------------
    def heptadiagonal(self, P_old):

        nt = self.field.grid.nt
        nx = self.field.grid.nx
        ny = self.field.grid.ny
        nz = self.field.grid.nz

        # Perms → SI
        Kx = self.field.properties.permx * md_to_m2
        Ky = self.field.properties.permy * md_to_m2
        Kz = self.field.properties.permz * md_to_m2

        Ax = self.field.grid.ax
        Ay = self.field.grid.ay
        Az = self.field.grid.az

        dx = self.field.grid.dx
        dy = self.field.grid.dy
        dz = self.field.grid.dz

        bo, uo = self.get_fluid_proprerties(P_old/kgf_cm2_to_Pa)
        uo = uo * cp_to_Pa_s    # cp → Pa.s
        gamma = self.get_gamma_term(P_old)

        # COO arrays
        rows, cols, data = [], [], []

        # inicializa diagonal apenas com Gamma
        diagonal = xp.array(gamma, dtype=float)

        ind_z, ind_y, ind_x = xp.indices((nz, ny, nx))
        all_nodes = xp.arange(nt)

        def harm(a, b):
            return 2.0 / (1.0/a + 1.0/b)

        # X
        mask_x = (ind_x < nx - 1).ravel()
        id_L = all_nodes[mask_x]
        id_R = id_L + 1

        Tx = harm(Kx[id_L], Kx[id_R]) * Ax[id_L] / (uo[id_L] * bo[id_L] * dx)

        rows += [id_L, id_R]
        cols += [id_R, id_L]
        data += [-Tx, -Tx]

        xp.add.at(diagonal, id_L, Tx)
        xp.add.at(diagonal, id_R, Tx)

        # Y
        mask_y = (ind_y < ny - 1).ravel()
        id_S = all_nodes[mask_y]
        id_N = id_S + nx

        Ty = harm(Ky[id_S], Ky[id_N]) * Ay[id_S] / (uo[id_S] * bo[id_S] * dy)

        rows += [id_S, id_N]
        cols += [id_N, id_S]
        data += [-Ty, -Ty]

        xp.add.at(diagonal, id_S, Ty)
        xp.add.at(diagonal, id_N, Ty)

        # Z
        mask_z = (ind_z < nz - 1).ravel()
        id_B = all_nodes[mask_z]
        id_T = id_B + nx * ny

        dist_z = 0.5 * (dz[id_B] + dz[id_T])
        Tz = harm(Kz[id_B], Kz[id_T]) * Az[id_B] / (uo[id_B] * bo[id_B] * dist_z)

        rows += [id_B, id_T]
        cols += [id_T, id_B]
        data += [-Tz, -Tz]

        xp.add.at(diagonal, id_B, Tz)
        xp.add.at(diagonal, id_T, Tz)

        rows = xp.concatenate(rows)
        cols = xp.concatenate(cols)
        data = xp.concatenate(data)

        return rows, cols, data, diagonal


    # ---------------------------------------------------------
    def build_matrix(self, rows, cols, data, diagonal):

        nt = self.field.grid.nt
        diag_idx = xp.arange(nt)

        A = csr_matrix((data, (rows, cols)), shape=(nt, nt))
        A = A + csr_matrix((diagonal, (diag_idx, diag_idx)), shape=(nt, nt))

        return A


    def build_rhs(self, gamma, P_old):
        return gamma * P_old


    # ---------------------------------------------------------
    # SIMULAÇÃO COMPLETA (tudo em PA)
    # ---------------------------------------------------------
    def simulate(self, P0):

        nt = self.field.grid.nt

        # P0 em Pa
        P_old = xp.ones(nt, dtype=float) * (P0 * kgf_cm2_to_Pa)

        t = 0.0
        t_max = self.simulation_time
        dt = self.time_step

        while t < t_max:
            t += dt

            rows, cols, data, diagonal = self.heptadiagonal(P_old)
            A = self.build_matrix(rows, cols, data, diagonal)

       
            gamma = self.get_gamma_term(P_old)

            b = self.build_rhs(gamma, P_old)
            
  
            P_new, info = cg(A, b, x0=P_old, tol=1e-5)

            if info != 0:
                print("Convergência falhou!")
                break

            P_old = P_new

            print(f'Propriedades    {      self.get_fluid_proprerties(P_old/kgf_cm2_to_Pa)}')

            print(f"Tempo de simulação: {t/day_to_sec:.2f} dias")
            print(f"Pressão média: {xp.mean(P_old)/kgf_cm2_to_Pa:.2f} kgf/cm²\n")
        

        # retorna para kgf/cm²
        return P_old / kgf_cm2_to_Pa
