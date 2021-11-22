import numpy as np
import scipy.integrate as itg
from matplotlib.pyplot import plot, show
from math import log, sqrt
from scipy.integrate import trapezoid

""" define some constant parameters """
Iyy = 0.0055 * 3 #moment of inertia, kg*m^2
cg_Ttheta = 0.209 #moment arm of thrust vector
m = 0.5508 #mass, kg
v_0 = 20 #initial downward velocity, m/s
theta_T = 5 * np.pi / 180 * 0.9 #angular thrust deflection, rad
TWR = 2.5 # thrust-to-weight, -
dt = 0.01
g = 9.81 #gravitational acceleration, m/s^2
A_0 = np.sin(theta_T) * cg_Ttheta / Iyy * TWR * m * g * 0.5 #angular acceleration, rad/s^2
print(A_0)
t_burn = sqrt(2*np.pi/A_0)
w_0 = A_0*t_burn/2

""" start to mid vertical accerlation function """
def T1(t):
    return 0.5*g*TWR*np.cos(np.pi/2 - 1/2 * A_0 * t**2 + theta_T) + 0.5*g*TWR*np.cos(np.pi/2 - 1/2 * A_0 * t**2)

""" mid to end vertical accerlation function """
def T2(t):
    return 0.5*g*TWR*np.cos((np.pi/4 + 1/2 * A_0 * t**2 - w_0*t - theta_T)) + 0.5*g*TWR*np.cos((np.pi/4 + 1/2 * A_0 * t**2 - w_0*t))

def returnIntegralList():
    m = []
    n = []
    cur_T = 0
    while cur_T < t_burn/2:
        m.append(cur_T)
        n.append(itg.quad(T1, 0, cur_T)[0])
        cur_T += dt
    next_T = t_burn/2
    while next_T < t_burn:
        m.append(next_T)
        n.append(itg.quad(T1, 0, t_burn/2)[0]+itg.quad(T2, t_burn/2, cur_T)[0])
        next_T += dt
    return m,n

x,y = returnIntegralList()

print(trapezoid(x,y,dt))
