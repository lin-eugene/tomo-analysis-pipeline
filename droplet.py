from math import *


R=7E-9  #m

psv = 0.73 # cm^3/ g - partial specific volume
rho=1/(psv*1E-6)/1E3   #kg/m^3



V=4./3.*pi*R**3

Mw=200*120  #Da
Mw=25429.94
conv=1.66054e-27  #1Da is xxx kG
Mass=V*rho #kg

n=Mass/(Mw*conv)
print(n)



#sig=0.1    #standard deviation of droplet gaussian density distribution.
#rho(r)=rho0*exp(-(r)**2/((2*sig**2)))
