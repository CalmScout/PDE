from input import generate_params_dict
from visualization import animateU
from visualization import plotLayerOfU
# from console.difference_schemes import explicit_cross_scheme
from difference_schemes import implicit_scheme
from difference_schemes import implicit_crank_nicolson_scheme

# input constants and functions strings from file
# file_url = 'files/input/test_input.txt'
file_url = 'files/input/quasilinear_02_parabolic_equation.txt'
p = generate_params_dict(file_url=file_url)


# U, xArgs, tArgs = explicit_cross_scheme(p)
# U, xArgs, tArgs = implicit_scheme(p)
U, xArgs, tArgs = implicit_crank_nicolson_scheme(p)

# # Run animation for
# animateU(U, xArgs, tArgs)

for j in range(len(tArgs)):
    plotLayerOfU(U, j, xArgs, p['t_min'], p['delta_t'])
    j += 1