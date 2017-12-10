"""
Contains functions for data input.
"""


def input_file(file_url):
    print("Input will be taken from a file:", file_url)

    def get_value(file):
        """
        Skips strings which are begins from '#'. File contains data in format:
        'name of param' = 'value'. Returns value like string.
        """
        temp_str = file.readline()
        while temp_str[0] == '#':
            temp_str = file.readline()
        return temp_str.split(sep='=')[1].strip()

    with open(file_url) as file:
        # scheme parameters
        t_min = float(get_value(file))
        t_max = float(get_value(file))
        x_min = float(get_value(file))
        x_max = float(get_value(file))
        delta_t = float(get_value(file))
        delta_x = float(get_value(file))
        # initial conditions
        init_cond_layer_0_str = get_value(file)
        init_cond_layer_1_str = get_value(file)
        init_cond_time_derivative_str = get_value(file)
        # boundary conditions
        boundary_left_str = get_value(file)
        boundary_right_str = get_value(file)
        # equation parameters
        tau = float(get_value(file))
        r_func_str = get_value(file)
        alpha = float(get_value(file))
        s_func_str = get_value(file)
        beta = float(get_value(file))
        phi_func_str = get_value(file)
        mu = float(get_value(file))
        k_func_str = get_value(file)
        nu = float(get_value(file))
        psi_func_str = get_value(file)
        gamma = float(get_value(file))
        h_func_str = get_value(file)
        xi = float(get_value(file))
        b_func_str = get_value(file)
        theta = float(get_value(file))
        f_func_str = get_value(file)
    return t_min, t_max, x_min, x_max, delta_t, delta_x, init_cond_layer_0_str, \
           init_cond_layer_1_str, init_cond_time_derivative_str, \
           boundary_left_str, boundary_right_str, tau, r_func_str, alpha, \
           s_func_str, beta, phi_func_str, mu, k_func_str, nu, psi_func_str, \
           gamma, h_func_str, xi, b_func_str, theta, f_func_str


def generate_params_dict(file_url):
    t_min, t_max, x_min, x_max, delta_t, delta_x, init_cond_layer_0_str,\
    init_cond_layer_1_str, init_cond_time_derivative_str, boundary_left_str,\
    boundary_right_str, tau, r_func_str, alpha, s_func_str, beta,\
    phi_func_str, mu, k_func_str, nu, psi_func_str, gamma, h_func_str, xi,\
    b_func_str, theta, f_func_str = input_file(file_url)

    params = {
        # scheme parameters
        't_min': t_min,
        't_max': t_max,
        'x_min': x_min,
        'x_max': x_max,
        'delta_t': delta_t,
        'delta_x': delta_x,
        # initial conditions
        'init_cond_layer_0_str': init_cond_layer_0_str,
        'init_cond_layer_1_str': init_cond_layer_1_str,
        'init_cond_time_derivative_str': init_cond_time_derivative_str,
        # boundary conditions
        'boundary_left_str': boundary_left_str,
        'boundary_right_str': boundary_right_str,
        # equation parameters constants and functions
        'tau': tau,
        'r_func_str': r_func_str,
        'alpha': alpha,
        's_func_str': s_func_str,
        'beta': beta,
        'phi_func_str': phi_func_str,
        'mu': mu,
        'k_func_str': k_func_str,
        'nu': nu,
        'psi_func_str': psi_func_str,
        'gamma': gamma,
        'h_func_str': h_func_str,
        'xi': xi,
        'b_func_str': b_func_str,
        'theta': theta,
        'f_func_str': f_func_str
    }

    return params
