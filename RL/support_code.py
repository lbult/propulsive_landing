import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

def linearFunc(x,a,b):
    return a*x + b

class Render():
    def __init__(self, t, data, actuator, reward):
        self.reward = reward # reward output, integer
        self.data = data # list of data entries, t, posx, posy, velx, vely, control
        self.t = t
        self.actuator = actuator

    def press(self, event):
        print('press', event.key)
        if event.key == 'c':
            print("Continue to next render")
        #if event.key == 'n':
        #    print("Has been moved to Not Shapiro")
    
    def Run(self):
        fig, axs = plt.subplots(2, 3)
        matplotlib.use('TkAgg')
        fig.suptitle("Reward: " + str(self.reward))
        # t-y
        axs[0, 0].plot(self.t, self.data[1])
        axs[0, 0].set_title('Position y vs time')
        #axs[0, 0].xlabel("x [m]")
        #axs[0, 0].xlabel("y [m]")  
        # t-vx
        axs[0, 1].plot(self.t, self.data[2], 'tab:orange')
        axs[0, 1].set_title('Velocity in x-direction')
        #axs[0, 1].xlabel("t [s]")
        #axs[0, 1].xlabel("Velocity x [m/s]")  
        # t-vy
        axs[1, 0].plot(self.t, self.data[3], 'tab:orange')
        axs[1, 0].set_title('Velocity in y-direction')
        #axs[1, 0].xlabel("t [s]")
        #axs[1, 0].xlabel("Velocity x [m/s]") 
        # t - actuator input
        axs[1, 1].plot(self.t, self.actuator, 'tab:red')
        axs[1, 1].set_title('Actuator Deflection vs time')
        #axs[1, 0].xlabel("t [s]")
        #axs[1, 0].xlabel("Actuator angle [rad]") 
        
        axs[1, 2].plot(self.t, self.data[4], 'tab:red')
        axs[1, 2].set_title('Vehicle angle vs time')

        axs[1, 2].plot(self.t, self.data[5], 'tab:red')
        axs[1, 2].set_title('Vehicle angular velocity vs time')
        
        plt.show()

class SRM():
    def __init__(self, engine_type, t, dt):
        self.engine_type = engine_type
        self.t = t # [s]
        self.dt = dt # discretisation, necessary for weight calculations
        self.thrust_force = 0 # [N]
        self.impulse = 0

        if self.engine_type ==  "1/2A6":
            self.engine_param = [8.55737293*10**-5,  2.16185198*10**0,  1.03338808*10**3, -3.27246718*10**4,
                    5.61765793*10**5, -4.58971034*10**6,  1.31742637*10**7,  4.09887150*10**7,
                    -3.07523806*10**8,  1.29853726*10**8,  1.93955863*10**9, -3.51515455*10**8,
                    -1.38401836*10**10,  1.86375875*10**10]
            self.t_burn = 0.32 # [s]
            self.total_impulse = 1.1 # [Ns]
            self.prop_weight = 0.0026 # [kg]
            self.dry_mass = 0.01 # check this

        if engine_type == "F-15-0":
            self.engine_t = [0, 0.148 ,0.228, 0.294, 0.353, 0.382, 0.419, 0.477, 0.520, 0.593, 0.688, 0.855, 1.037,
                            1.205,1.423,1.503,1.736,1.955,2.210,2.494,2.763,3.120,3.382 ,3.404 ,3.418 ,3.450]
            self.engine_y = [0, 7.638, 12.253, 16.391, 20.210, 22.756, 25.260, 23.074, 20.845, 19.093, 17.5, 
                            16.255, 15.427, 14.448, 14.627, 14.758, 14.623, 14.303, 14.141, 13.819,
                            13.338, 13.334, 13.013, 9.352, 4.895, 0.000]
            self.t_burn = 3.44 # [s]
            self.total_impulse = 49.6 # [Ns]
            self.prop_weight = 0.06 # [kg]
            self.dry_mass = 0.1 # [kg]
            self.total_mass = self.prop_weight + self.dry_mass # [kg]

            # assume fuel spent is proportion
            # total_weight = 100g
            # https://www.thrustcurve.org/motors/Estes/F15/

    """ Thrust Curve Code (SRM)"""
    def engineCurve(self,x,a,b,c,d,e,f,g,h,i,j,k,l,m,n):
        if self.engine_type ==  "1/2A6":
            return a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 +  h*x**7 + i*x**8 +  j*x**9 + k*x**10 +  l*x**11 + m*x**12 +  n*x**13
        # piecewise-linear function, maybe a cubic spline in a future iteration
        # see https://stackoverflow.com/questions/46637605/difference-between-quadratic-and-2nd-order-spline-interpolation-in-scipy
        if self.engine_type == "F-15-0":
            prev_t = 0
            prev_y = 0
            for i,j in zip(self.engine_t,self.engine_y):
                if self.t < i and self.t >= prev_t:
                    param, = curve_fit(linearFunc, [prev_t, i], [prev_y, j])
                    self.thrust_force = linearFunc(self.t, *param)
                    self.impulse += self.thrust_force * self.dt
                    self.total_mass = self.dry_mass + 0.06 * (1-self.impulse/self.total_impulse)
                prev_t = i
                prev_y = j
