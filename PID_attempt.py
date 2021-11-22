from typing import ValuesView
import numpy as np
import scipy as sc
from matplotlib.pyplot import plot, show, scatter
from math import log, pi, cos
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.sparse.extract import find

""" Thrust Curve Code (SRM)"""
def engineCurve(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12 +  n*x**13

def Thrust(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    if x > 0.02 and x < 0.32:  
        return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12 +  n*x**13
    else:
        return 0

def dEngineCurvedx(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    #return a*x + 1/2*b*x**2 + 1/3*c*x**3 + 1/4*d*x**4 + 1/5*e*x**5 + 1/6*f*x**6 +  1/7*g*x**7 + 1/8*h*x**8 +  1/9*i*x**9 + 1/10*j*x**10 +  1/11*k*x**11 + 1/12*l*x**12 +  1/13*m*x**13 +  1/14*n*x**14
    return 1/2*a*x**2 + 1/3*1/2*b*x**3 + 1/4*1/3*c*x**4 + 1/4*1/5*d*x**5 + 1/6*1/5*e*x**6 +  1/6*1/7*f*x**7 + 1/8*1/7*g*x**8 +  1/8*1/9*h*x**9 + 1/9*1/10*i*x**10 +  1/10*1/11*j*x**11 + 1/11*1/12*k*x**12 +  1/12*1/13*l*x**13 +  1/13*1/14*m*x**14 + 1/14*1/15*n*x**15

x_datp = [0,54,27,81,108,138,162,183,206,226,240,258,270,284,293,304,315,326,336,347,356]
x_dat = [element *(0.05/54) for element in x_datp]
y_datp = [0,17,7,32,49,71,90,109,126,139,142,126,108,85,71,53,37,22,12,5,0]
y_dat = [element *(2/38) for element in y_datp]

param, cov = curve_fit(engineCurve, x_dat, y_dat)

x = np.arange(0.02,0.32,0.001)
y = engineCurve(x, *param)
y_integral = dEngineCurvedx(x, *param)

#scatter(x_dat,y_dat)
#plot(x,y)
#show()

dx = 0.004
I_est = 8*trapezoid(y,x,dx)

"""Control Code"""

dt = 0.0005
te_nom = max(y_dat)
az = [0]
vz = [-10]
z = [10]
m = 0.52
sburn = 0.30
g = 9.81
h_est = 0.1

def Controller(I_est, I_req, t_burn, total_t):
    K_d = 0.030
    K_t = 0.03
    #if I_est-I_req > 0:
    theta = K_d * abs(I_est-I_req) - K_t*((total_t-t_burn)*1)**2
    #$else:
        #theta = 0
    if theta < 15*pi/180:
        return theta
    else:
        return 15*pi/180

print("Available momentum: " + str(I_est) + " [N s]")

T = 0
t_list = [0]
running = True
T_burn = 0
calcNew = True

xs2 = np.arange(0,0.02,dx)
ys2 = engineCurve(xs2, *param)
left_bound = 8*trapezoid(ys2,xs2,dx)/m
xs2 = np.arange(0,0.32,dx)
ys2 = engineCurve(xs2, *param)
right_bound = 8*trapezoid(ys2,xs2,dx)/m

while running:
    T += dt
    t_list.append(T)
    if calcNew:
        findT = True
        tspecial = 0.1
        while findT:
            xs = np.arange(0,tspecial,dx)
            ys = engineCurve(xs, *param)
            v_end = -8*trapezoid(ys,xs,dx)/m + left_bound + g*(tspecial-0.02) - (vz[-1])
            if tspecial > 0.33:
                h_est = 0.1
                findT = False
            if v_end < 0.5 and v_end > -0.7:
                h_est = -vz[-1]*(tspecial) + 1/2 * 9.81 * (tspecial)**2 - 8*dEngineCurvedx((tspecial),*param)/m + 8*dEngineCurvedx(0.02,*param)/m + 0.025
                findT = False
            tspecial += 0.01
    if z[-1] < h_est and z[-1]  > 0: #z[-1] < h_est:
        delta_vz = vz[-1]-vz[-2]
        xs = np.arange(0,T_burn,dx)
        ys = engineCurve(xs, *param)
        I_est = right_bound - 8*trapezoid(ys2,xs2,dx)/m
        F = 8*Thrust(T_burn, *param)
        commanded_angle = Controller(I_est, m*abs(vz[-1]), T_burn, 0.3)
        az.append(F/m * cos(commanded_angle)-9.81)
        T_burn += dt
        print(commanded_angle)
        calcNew = False
    else:
        az.append(-9.81)
        #print(h_est)

    vz.append(vz[-1]+az[-1]*dt) 
    z.append(z[-1]+vz[-1]*dt)#+1/2*az[-1]*dt**2)
    if z[-1] <= -0.005:
        running = False
    if T>2:
        running = False
    #if T_burn > 0.30:
        #running = False

print("Start of burn: " + str(h_est) + " [m]")
print("Required momentum: " + str(m*vz[0]) + " [N s]")
print("Velocity at ground strike: " + str(vz[-1]))
#plot(t_list,vz)
plot(t_list,z)
#plot(t_list,az)
show()
    