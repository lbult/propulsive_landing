import numpy as np
import scipy.integrate as integrate
from matplotlib.pyplot import plot, show
from math import log

g = 9.80665
m = 1
m_dot = 0.05
F = 10

tb = 10
dt = 0.01

t = np.arange(0, tb, dt)

a = F/(m-m_dot*t) - g
v = -F/m_dot * np.log(m-m_dot*t) - g*t

print(integrate.trapezoid(v,t,dt))

s = -F/m_dot * ((t-m/m_dot) * np.log(m-m_dot*t) - t + m/m_dot) - g * t**2 /2 + 4000

a2 = []
v2 = [-40.562936111989075]
s2 = [123]
mnew = m
count = 0

for i in t:
    count += 1
    mnew -= m_dot*dt
    a2.append(F/(mnew) - g)
    v2.append(a[-1]*dt + v2[-1])
    s2.append(a[-1]*dt**2 / 2 + v2[-1]*dt + s2[-1])

print(count)
#t = np.arange(0, tb, dt)   
#print(integrate.trapezoid(v2,t))

plot(v)
plot(v2)
show()