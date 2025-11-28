from engine_grid import *
from engine_properties import *
from engine_fluid import *


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

print(pvt.pb)

grid = StructuredGrid.from_file('grid.txt')

properties = ReservoirProperties.from_file('petro.INC')



print(pvt.get_pvt_properties(500))
properties.validate_with_grid(grid)
# print("Simulação pronta para iniciar.")
a =1

