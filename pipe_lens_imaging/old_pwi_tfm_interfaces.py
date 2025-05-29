import numpy as np



def G2(xe, xa, za, c1, c2, theta, f, dfdx):
    A = xe - xa
    B = f(xe) - za
    D = np.sqrt(A**2 + B**2)
    fp = dfdx(xe)

    numerator_left = A + fp * B
    denominator_left = c1 * D

    numerator_right = np.sin(theta) + fp * np.cos(theta)
    denominator_right = c2

    return numerator_left / denominator_left - numerator_right / denominator_right

def G(xe, xa, za, c1, c2, theta, f, dfdx):
    ze = f(xe)
    den = np.sqrt((xe - xa)**2 + (ze - za)**2)
    termA = dfdx(xe) * (c2/c1 * (ze - za) / den - np.cos(theta))
    termB = c2/c1 * (xe - xa) / den - np.sin(theta)
    return termA + termB


def cost_fun(xe, xa : float, za : float, c1, c2, f, dfdx, steering_ang):
    fun = G(xe, xa, za, c1, c2, steering_ang, f, dfdx)
    return fun**2

# TODO: derivar G(x) para dfdx(x) e f(x) sendo equações exatas do círculo.

def cost_fun_prime(xe, xa : float, za : float, c1, c2, f, dfdx, steering_ang, deltax=1e-6):
    fun0 = cost_fun(xe, xa, za, c1, c2, f, dfdx, steering_ang)
    fun = cost_fun(xe + deltax, xa, za, c1, c2, f, dfdx, steering_ang)
    dfdx = (fun - fun0)/deltax
    return dfdx