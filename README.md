# Compton-scattering_monte-carlo
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:47:36 2020

@author: arpit
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from random import random,seed
import scipy.interpolate
import scipy.integrate
c=3e8      #velocity of light in m/s
tau=0.001  #optical depth of plasma cloud
R=2e10     #Radius of plasma cloud in m
N=30000    #Number of Photons
n=np.arange(0,N,1)

def F(x):  #analytical approximation of F(x)
    if x>=0 and x<=0.5:
        return (x**2/6.0)+0.047*x**3-0.03*x**4+(x**2/(2*(1+x)))
    elif x>=0.5 and x<=3.5:
        return (1+x)*np.log(1+x)-0.94*x-0.00925
    else :
        return (1+x)*np.log(1+x)-0.5*x-13.16*np.log(2+0.076*x)+9.214

def I(z):  # the integration is for CPDF of rho1(x|v)
    sum=scipy.integrate.romberg(Rho1,0.00001,z)
    return sum

def Ne(beta):  #velocity distribution function for electrons (nonrelativistic maxwellian distribution)

    K=511.0/(25*2)   #k=mc^2/2kTe so kTe=25kev
    f=beta**2*math.exp(-K*beta**2)    # here beta=v/c
    return f

def Rho1(x):   #pdf for modelling x
    hv=5.11e-6   # energy of photon in kev
    m=511        #rest mass energy of electron
    k=511.0/(25*2)          #k=mc^2/2KTe
    x1=(2*hv/m)*math.sqrt((1-x)/(1+x))
    x2=(2*hv/m)*math.sqrt((1+x)/(1-x))
    gamma=math.sqrt(1.0/(1-x*x))  #lorentz factor
    return Ne(x)*(F(x2)-F(x1))*1.0/(gamma*math.sqrt(gamma**2-1))        

def arcF(m):     #analytical approximation of x where arcF(m)=inverse of F=x 
    if m<6e-4 :
        return math.sqrt(1.5*m)
    elif m>6e-4 and m<2.311e-2 :
        return 0.5+math.cos(1/3*math.acos(1-9*m)+4*np.pi/3.0)
    else :
        t=np.log(m)-np.log(F(10**(-0.5)))
        return 10**(-0.5+0.228*t+4.708e-3*t**2+3.114e-4*t**3-1.921e-5*t**4+2.873e-7*t**5)

n1=[]   
hv2=[]
CPDF=[]
beta= np.linspace(0.000001,0.99999,1000)
V=beta*c
for m in beta : #linear interpolation for getting CPDF of rho1(x|v)
    CPDF.append(I(m)/(I(0.9999)))
y_interp = scipy.interpolate.interp1d(CPDF, V)

for i in range (N): 
    theta=np.pi*random()  # Choosing different theta and phi for every photon
    phi0=2*np.pi*random()
    omegax=np.cos(phi0)*np.sin(theta)    # omega's are photon direction before Scateering
    omegay=np.sin(phi0)*np.sin(theta)
    omegaz=np.cos(theta)
    x,y,z=0,0,0 # Starting point for Photons (central photon source)
    w=1   # photon weight
    epsilon=1e-4    #photons with smaller than this weight will simply escape (Wmin)
    hv=1e-8   #delta function for photon energy in units of mc^2
    count=0 #for measuring average number of scttering
    while w>epsilon:
        
        r=np.sqrt(x**2+y**2+z**2)  # distance from center
        rpsi=(x*omegax+y*omegay+z*omegaz) #dot product of r and psi
        l=-rpsi+np.sqrt(R**2-r**2+rpsi**2) # distance between boundry and the point of collision 
        w=w*(1-np.exp(-tau*l/R))           # updating w after each collision
        lam=-(R/tau)*np.log(1-random()*(1-np.exp(-tau*l/R))) # photon free path
        if lam>l:
            print('particle escapes')   # if lamda is greater than l, particle escapes
            break
        x+=lam*omegax
        y+=lam*omegay       #if particle does not escapes, these are components of new position vector for collision 
        z+=lam*omegaz
        C=random()
        v=y_interp(C)       #velocity of colliding electron
        x1=(2*hv)*math.sqrt((1-(v/c))/(1+(v/c)))
        x2=(2*hv)*math.sqrt((1+(v/c))/(1-(v/c)))   # x1,x2 are limits for finding x
        f=F(x1)+random()*(F(x2)-F(x1))            #Value of CPDF for x
        g=math.sqrt(1.0/(1-(v/c)**2))             #lorentz factor
        u=(c/v)*(1-(arcF(f)/(2*hv*g)))            #u =direction (cos(theta)) of photon before Scattering
        if u>1 :                         
            u=0.9999
        if u<-1 :           # kepping the u's which are exceeding the limit due to error in analytical approximation of x.
            u=-0.9999
        phi=2*np.pi*random() #modelling of P(phi)
        vx=v*(omegax*u-np.sqrt((1-u**2)/(1-omegaz**2))*(omegax*omegaz*np.cos(phi)+omegay*np.sin(phi)))  
        vy=v*(omegay*u-np.sqrt((1-u**2)/(1-omegaz**2))*(omegay*omegaz*np.cos(phi)-omegax*np.sin(phi))) # vx,vy,vz are component of velocities of electron before scattering
        vz=v*(omegaz*u+(np.sqrt((1-u**2)*(1-omegaz**2)))*np.cos(phi))
        vx0,vy0,vz0=vx/v,vy/v,vz/v
        cnt=0               # checking for infinite loop
        while True :
                        if cnt>10000:
                             print ("stuck in loop")
                             break
                        cnt += 1
                        G1=random() 
                        phi1=2*np.pi*G1   
                        G2=random()    # G1 and G2 for modelling direction of photon after scattering 
                        u1=((v/c)+2*G2-1)/(1+(v/c)*(2*G2-1))  # direction (cos(theta)) of photon after Scattering 
                        omegax1=vx0*u1-np.sqrt((1-u1**2)/(1-vz0**2))*(vx0*vz0*np.cos(phi1)+vy0*np.sin(phi1)) #transformation from spherical to cartesian coordinate
                        omegay1=vy0*u1-np.sqrt((1-u1**2)/(1-vz0**2))*(vy0*vz0*np.cos(phi1)-vx0*np.sin(phi1)) # omega1's are direction of photons(after Scattering) in cartesian coordinate
                        omegaz1=vz0*u1+(np.sqrt((1-u1**2)*(1-vz0**2)))*np.cos(phi1)
                        x3=2*(hv)*g*(1-(v/c)*u)     # value of x before scattering
                        hv1=hv*(1-(vx*omegax+vy*omegay+vz*omegaz)/c)/(1-(v/c)*u1+(hv/(g))*(1-(omegax*omegax1+omegay*omegay1+omegaz*omegaz1))) # new energy of photon due to scattering
                        x4=2*(hv1)*g*(1-(v/c)*u1)    # value of x after Scattering
                        X=(x3/x4)+(x4/x3)+4*((1/x3)-(1/x4))+4*((1/x3)-(1/x4))**2  # for modelling rho4(u'|v)
                        Y=X*(x4/x3)**2   
                        G3=random()
                        if G3<(Y/2) : # rejection technique Y<2
                            break
        
        # hv1 is energy of photon after scattering 
        hv1=hv*(1-(vx*omegax+vy*omegay+vz*omegaz)/c)/(1-(v/c)*u1+(hv/(g))*(1-(omegax*omegax1+omegay*omegay1+omegaz*omegaz1)))
        omegax,omegay,omegaz=omegax1,omegay1,omegaz1 # updating the omega's after scattering
        hv=hv1 # updating the energy
        count+=1
        
        
    hv2.append(hv) #energy of photon after escape    
    n1.append(count)
data = np.column_stack((n, hv2))
Ns=0
for i in range(N): Ns+=n1[i]
print('Average No of Scattering = ',Ns/N)
np.savetxt('montec.txt', data)
hv3=np.loadtxt('montec.txt')  
plt.figure(figsize=(10,5))
ax=plt.axes(xscale='log',yscale='log')
plt.hist(hv3[:,1],histtype='step',bins=np.logspace(np.log10(hv3[:,1].min()),np.log10(hv3[:,1].max()),35),linewidth=2)
plt.xlabel(r'Photon Energy($M_{E}c^2$)')
plt.ylabel('Number of Photons')
plt.title('Radiation Spectrum through Comptonization')
plt.show()
