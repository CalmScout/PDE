import numpy as np
import matplotlib.pyplot as plt
from console.input import generate_params_dict
from console.visualization import animateU
from console.visualization import plotLayerOfU
from console.difference_schemes import explicit_cross_scheme
from console.difference_schemes import implicit_scheme
from console.difference_schemes import implicit_crank_nicolson_scheme

# input constants and functions strings from file
file_url = 'files/input/parabolic_01.txt'
p = generate_params_dict(file_url=file_url)


# u_explicit, _, _ = explicit_cross_scheme(p)
u_implicit, _, _ = implicit_scheme(p)
u_crank_nicolson, _, _ = implicit_crank_nicolson_scheme(p)
# print('u_implicit.shape',u_implicit.shape)
# print('u_crank_nicolson.shape',u_crank_nicolson.shape)
print('len(u_implicit)', len(u_implicit))
print('len(u_crank_nicolson)', len(u_crank_nicolson))



def calculate_layer_discrepancy(u, v, j):
    """
    Calculates sum of abs of element-wise differences of 'j'th layer of matrices
    'u', 'v'
    """
    discrepancy = 0.0
    for i in range(len(u)):
        discrepancy += np.abs(u[i][j] - v[i][j])
    return discrepancy

discr_vec = np.zeros(len(u_crank_nicolson[0]))
for j in range(len(u_crank_nicolson[0])):
    discr_vec[j] = calculate_layer_discrepancy(u_crank_nicolson, u_implicit,j)

plt.plot(discr_vec, 'ro')
plt.show()