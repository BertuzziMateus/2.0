from engine_grid import *
from engine_properties import *
from engine_fluid import *
from engine_reservoir import *
from engine_simulation import *

# nt = 81*58*20


# grid = StructuredGrid(
#     x_length = 8100,
#     y_length = 5800,
#     thickness = [
#         8.56,8.56,8.56,8.56,8.56,
#         8.56,8.56,8.56,8.56,8.56,
#         8.56,8.56,8.56,8.56,8.56,
#         8.56,8.56,8.56,8.56,8.56
#         ],
#     nx = 81,
#     ny = 58,
#     nz = 20,
#     actnum = xp.ones(81*58*20)
# )



# properties = ReservoirProperties(
#     porosity = xp.ones(nt)*0.2,
#     permx = xp.ones(nt)*100,
#     permy = xp.ones(nt)*100,
#     permz = xp.ones(nt)*100*0.1,
#     ntg = xp.random.rand(nt)*1
# )



# pvt = Fluid(
#     pressures=[1,360],
#     bo= [1,1.5],
#     u_o=[0.3,1.2],
#     pb = 190

# )




pvt = Fluid.from_file('pvt_campo.csv', bubble_point = 190.0)

grid = StructuredGrid.from_file('grid.txt')

properties = ReservoirProperties.from_file('petro.INC')

field_ = Field(grid, properties, pvt,5.4045e-5)


sim = SimulationEngine(
    field_,
    [],
    1000,
    1,
)








P = sim.simulate(P0=200)   # pressão inicial em kgf/cm2
print(P)

# print(pvt.get_pvt_properties(500))
# properties.validate_with_grid(grid)
# print("Simulação pronta para iniciar.")


