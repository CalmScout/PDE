"""
Contains difference schemes for PDEs.
"""
import numpy as np
from math import *
from pylab import *


def __define_functions(p):
    """
    Defines functions from the dict and return them
    :param p: dict which contains functions like strings.
    :return: tuple of functions
    """

    # functions for scheme creation
    # initial condition functions - 0 and 1st layers
    def init_cond_layer_0(x):
        return eval(p['init_cond_layer_0_str'])

    def init_cond_layer_1(x):
        return eval(p['init_cond_layer_1_str'])

    def init_cond_time_derivative(x):
        return eval(p['init_cond_time_derivative_str'])

    # boundary conditions
    def boundary_left(t):
        return eval(p['boundary_left_str'])

    def boundary_right(t):
        return eval(p['boundary_right_str'])

    # functions for our equation definition
    def r(u):
        return eval(p['r_func_str'])

    def s(u):
        return eval(p['s_func_str'])

    def phi(u):
        return eval(p['phi_func_str'])

    def k(u):
        return eval(p['k_func_str'])

    def psi(u):
        return eval(p['psi_func_str'])

    def h(u):
        return eval(p['h_func_str'])

    def b(u):
        return eval(p['b_func_str'])

    def f(u):
        return eval(p['f_func_str'])

    return init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative,\
           boundary_left, boundary_right, r, s, phi, k, psi, h, b, f


def _create_scheme(p):
    """
    Creates necessary objects for schema work: grid, variables, functions
    :param p: dictionary of parameters
    :return: tuple of objects for schema work
    """
    # number of columns and rows in U table
    uTableXSize = int(ceil((p['x_max'] - p['x_min']) / p['delta_x']))
    uTableTSize = int(ceil((p['t_max'] - p['t_min']) / p['delta_t']))
    # vectors of arguments
    xArgs = np.zeros(uTableXSize)
    for i in range(len(xArgs)):
        xArgs[i] = p['x_min'] + i * p['delta_x']
    tArgs = np.zeros(uTableTSize)
    for j in range(len(tArgs)):
        tArgs[j] = p['t_min'] + j * p['delta_t']

    # creation of U matrix
    u = np.zeros((uTableXSize, uTableTSize))

    init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative, \
    boundary_left, boundary_right, r, s, phi, k, psi, h, b, f = \
    __define_functions(p)

    # fill in first and second layers
    for i in range(len(u)):
        u[i][0] = init_cond_layer_0(xArgs[i])
    for i in range(len(u)):
        u[i][1] = u[i][0] + p['delta_t'] * init_cond_time_derivative(xArgs[i])

    # boundary Condition
    for j in range(len(u[0])):
        u[0][j] = boundary_left(tArgs[j])
    for j in range(len(u[0])):
        u[-1][j] = boundary_right(tArgs[j])

    def pow2(x):
        return x * x

    return u, xArgs, tArgs, init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative,\
           boundary_left, boundary_right, r, s, phi, k, psi, h, b, f, pow2


def _create_system_of_linear_equations(u, j, coeff_m1, coeff, coeff_p1,
                                       b_current):
    """
    Constructs system of linear algebraic equations
    :param u: numerical solution of PDE, method construct system to fill in
    time layer 'j'
    :param j: current time layer, second coordinate of 'u' matrix
    :param coeff_m1: function for calculation element left to diagonal element
    :param coeff: function for calculation diagonal element
    :param coeff_p1: function for calculation element right to main diagonal
    :param b_current: function for calculating element at right equation part
    :return: tuple (a, b) - square matrix and right part for linear algebraic
    equation  
    """
    a_matrix = np.zeros((u.shape[0], u.shape[0]))
    #  'number of elements in space dimension' - square matrix
    b_right_part = np.zeros(u.shape[0])
    # for each internal element in the 'j'th time slice
    for i in range(1, u.shape[0] - 1):
        a_matrix[i][i - 1] = coeff_m1(u, i, j)
        a_matrix[i][i] = coeff(u, i, j)
        a_matrix[i][i + 1] = coeff_p1(u, i, j)
        b_right_part[i] = b_current(u, i, j)
    # fill in boundary conditions - left one
    a_matrix[0][0] = 1.0
    b_right_part[0] = u[0][j]
    # fill in boundary conditions - right one
    a_matrix[-1][- 1] = 1
    b_right_part[-1] = u[-1][j]

    return a_matrix, b_right_part


def _calculate_u_implicit(u, coeff_m1, coeff, coeff_p1, b_current):
    """
    Fill in 'u' solution matrix, using implicit approach.
    :return: u
    """
    # solve system with 0-iteration approximation
    for j in range(2, len(u[0])):
        a_mat, b_vec = _create_system_of_linear_equations(u, j, coeff_m1,
                                                          coeff, coeff_p1,
                                                          b_current)
        layer_solution = np.linalg.solve(a_mat, b_vec)
        # fill in 'u' matrix solution 'layer_solution'
        for i in range(len(u)):
            u[i][j] = layer_solution[i]
        print('Layer {} from {}'.format(j, len(u[0])))
    return u


def explicit_cross_scheme(p):
    """
    :param p: dictionary of parameters
    """
    u, xArgs, tArgs, init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative, \
    boundary_left, boundary_right, r, s, phi, k, psi, h, b, f, pow2 = \
        _create_scheme(p)
    while True:
        j = 2
        while j < len(tArgs):
            i = 1
            while i < len(xArgs) - 1:
                # check convergence and stability condition
                if p['delta_t'] > (0.5) * pow2(p['delta_x']) / (p['mu'] * k(u[i][j])):
                    raise ValueError("Doesn't meet convergence and stability condition")
                u[i][j] = u[i][j-1] + (p['delta_t'] * p['mu'] * k(u[i][j-1]) /
                                       pow2(p['delta_x'])) * (u[i+1][j-1] - 2*u[i][j-1] + u[i-1][j-1])
                i += 1
            print('Layer {} from {}'.format(j, len(u[0])))
            j += 1
        break
    print()
    return u, xArgs, tArgs


def implicit_scheme(p):
    """
    Solve PDE
    :param p: dictionary of parameters
    :return: tuple of coefficients matrix 'u' and lists of arguments 'xArgs', 'tArgs'
    """

    u, xArgs, tArgs, init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative, \
    boundary_left, boundary_right, r, s, phi, k, psi, h, b, f, pow2 = \
        _create_scheme(p)

    # weights of different layers input: sigma1, sigma2 in [0,1]
    sigma1 = 0.3
    sigma2 = 0.3
    sigma01 = 0.5

    def coeff_m1(u, i, j):
        return - p['mu'] * k(u[i][j-1]) / pow2(p['delta_x'])

    def coeff(u, i, j):
        return 1 / p['delta_t'] + 2 * p['mu'] * k(u[i][j-1]) / pow2(p['delta_x'])

    def coeff_p1(u, i, j):
        return - p['mu'] * k(u[i][j-1]) / pow2(p['delta_x'])

    def b_current(u, i, j):
        return u[i][j-1] / p['delta_t']

    _calculate_u_implicit(u, coeff_m1, coeff, coeff_p1, b_current)

    return u, xArgs, tArgs


def implicit_crank_nicolson_scheme(p):
    """
        Solve using Crank-Nicolson method PDE
        :param p: dictionary of parameters
        :return: tuple of coefficients matrix 'u' and lists of arguments 'xArgs', 'tArgs'
    """
    u, xArgs, tArgs, init_cond_layer_0, init_cond_layer_1, init_cond_time_derivative, \
    boundary_left, boundary_right, r, s, phi, k, psi, h, b, f, pow2 = \
        _create_scheme(p)

    sigma1 = 0.5

    def coeff_m1(u, i, j):
        return - p['mu'] * k(u[i][j-1]) * (1 - sigma1) / pow2(p['delta_x'])

    def coeff(u, i, j):
        return (1/p['delta_t'] + 2 * p['mu'] * k(u[i][j-1]) * (1-sigma1) / pow2(p['delta_x']))

    def coeff_p1(u, i, j):
        return - p['mu'] * k(u[i][j-1]) * (1 - sigma1) / pow2(p['delta_x'])

    def b_current(u, i, j):
        return (p['mu'] * k(u[i][j-1]) * sigma1 / pow2(p['delta_x'])) * u[i-1][j-1] + \
                        (1 / p['delta_t'] - 2 * p['mu'] * k(u[i][j-1]) * sigma1 / pow2(p['delta_x'])) * u[i][j-1] + \
                        (p['mu'] * k(u[i][j-1]) * sigma1 / pow2(p['delta_x'])) * u[i+1][j-1]



    _calculate_u_implicit(u, coeff_m1, coeff, coeff_p1, b_current)

    return u, xArgs, tArgs