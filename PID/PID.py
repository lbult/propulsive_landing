from typing import ValuesView
import math
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import log, pi, cos
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.sparse.extract import find

""" Thrust Curve Code (SRM)"""
def engineCurve(x,a,b,c,d,e,f,g,h,i,j,k,l,m):#,n):#,o,p,q,r,s,t,u,v,w,jj,y,z):
    return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12 #+  n*x**13 #+ o*x**14 + p*x**15 + q*x**16 + r*x**17 + s*x**18 + t*x**19 + u*x**20 + v*x**21 + w*x**22 + jj*x**23 + y*x**24 + z*x**25

def Thrust(x,a,b,c,d,e,f,g,h,i,j,k,l,m):
    if x > 0.02 and x < 0.32:  
        return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12
    else:
        return 0

def dEngineCurvedx(x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
    #return a*x + 1/2*b*x**2 + 1/3*c*x**3 + 1/4*d*x**4 + 1/5*e*x**5 + 1/6*f*x**6 +  1/7*g*x**7 + 1/8*h*x**8 +  1/9*i*x**9 + 1/10*j*x**10 +  1/11*k*x**11 + 1/12*l*x**12 +  1/13*m*x**13 +  1/14*n*x**14
    return 1/2*a*x**2 + 1/3*1/2*b*x**3 + 1/4*1/3*c*x**4 + 1/4*1/5*d*x**5 + 1/6*1/5*e*x**6 +  1/6*1/7*f*x**7 + 1/8*1/7*g*x**8 +  1/8*1/9*h*x**9 + 1/9*1/10*i*x**10 +  1/10*1/11*j*x**11 + 1/11*1/12*k*x**12 +  1/12*1/13*l*x**13 +  1/13*1/14*m*x**14 + 1/14*1/15*n*x**15

# For 1/2A6 engine
#x_datp = [0,54,27,81,108,138,162,183,206,226,240,258,270,284,293,304,315,326,336,347,356]
#x_dat = [element *(0.05/54) for element in x_datp]
#y_datp = [0,17,7,32,49,71,90,109,126,139,142,126,108,85,71,53,37,22,12,5,0]
#y_dat = [element *(2/38) for element in y_datp]

# For F-15-0 motor
x_datp = [0, 0.148 ,0.228, 0.294, 0.353, 0.382, 0.419, 0.477, 0.520, 0.593, 0.688, 0.855, 1.037,
          1.205,1.423,1.503,1.736,1.955,2.210,2.494,2.763,3.120,3.382 ,3.404 ,3.418 ,3.450]
y_datp = [0, 7.638, 12.253, 16.391, 20.210, 22.756, 25.260, 23.074, 20.845, 19.093, 17.5, 
            16.255, 15.427, 14.448, 14.627, 14.758, 14.623, 14.303, 14.141, 13.819,
            13.338, 13.334, 13.013, 9.352, 4.895, 0.000]

x_1 = x_datp[:int(len(x_datp)/2)]
x_2 = x_datp[int(len(x_datp)/2):]
y_1 = y_datp[:int(len(x_datp)/2)]
y_2 = y_datp[int(len(x_datp)/2):]

param, cov = curve_fit(engineCurve, x_1, y_1)
param2, cov2 = curve_fit(engineCurve, x_2, y_2)

#x = np.arange()
#y = engineCurve(x, *param)
#y_integral = dEngineCurvedx(x, *param)

'''
plt.plot(x_datp,y_datp)
plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")
plt.show()'''

dx = 0.004
I_est = 8*trapezoid(y_datp,x_datp,dx)

"""Control Code"""

dt = 0.0005
te_nom = 1.09*max(y_datp)
az = [0]
vz = [-13.0]
z = [10]
m = 1.735
sburn = 0.30
g = 9.81
h_est = 0.1

def Controller(I_est, I_req, t_burn, total_t):
    K_d = 0.030
    K_t = 0.03
    #if I_est-I_req > 0:
    theta = K_d * abs(I_est-I_req) - K_t*((total_t-t_burn))**2
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
            if v_end < 1.0 and v_end > -1.0:
                h_est = -vz[-1]*(tspecial) + 1/2 * 9.81 * (tspecial)**2 - 8*dEngineCurvedx((tspecial),*param)/m + 8*dEngineCurvedx(0.02,*param)/m + 0.025
                findT = False
            tspecial += 0.01
            h_est = 2.53
            findT = False
            calcNew = False
    if z[-1] < h_est and z[-1]  > 0: #z[-1] < h_est:
        delta_vz = vz[-1]-vz[-2]
        xs = np.arange(0,T_burn,dx)
        ys = engineCurve(xs, *param)
        I_est = right_bound - 8*trapezoid(ys2,xs2,dx)/m
        F = 8*1.09*Thrust(T_burn, *param)
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
#print("Required momentum: " + str(m*vz[0]) + " [N s]")
print("Velocity at ground strike: " + str(vz[-1]))

for i, j in enumerate(az):
    az[i] = j/9.81

#plt.plot(t_list,vz)

plt.plot(t_list,z)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.show()

plt.plot(t_list,az)
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [g]")
plt.show()
