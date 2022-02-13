# Author: Jesse Randall
# Advisor: Dr. Cooney
# Project follows Lambda-CDM Model of cosmology and units are in mks

'''
Edits:
04/07/18 changed units of c to km/s to match units in paper fixing initial error 
         in graph. Still is off by e3.
        
04/09/18 Added normalization that fixed the error.

10/08/18 Started working on project again. Trying to numerically integrate the 
         function for the variance on page 48 to graph the standard deviation
         as a function of M.
        
03/25/19 Haven't updated this edit log in a while. Finished section numerically
         integrating variance, however, it is done improperly and still needs to 
         be reworked. Started work on next section which involves integration
         of contributions to power spectrum. 
        
01/18/22 Continuation of the project after extended absence. Going over code now
         and am making a review of the progress we made so far.
'''

##### PACKAGES #####
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':16})

##### GLOBAL CONSTANTS #####

##### Plots #####
# 1 true, 0 false.
j           = 0    # Figure index. Increment when plotting to designate next figure.
FIGX        = 10   # Figure x dimension.
FIGY        = 8    # Figure y dimension.
POINTS      = int(1e4) # Number of points to plot.
# To plot something, change the 0 to a 1 for the plot you wish to make.
PLOTDLPS    = 1    # Plot dimensionless linear power spectrum of density fluctuations.
PLOTVARINT  = 1    # Plot integrand of variance of density fluctuations.
PLOTSD      = 1    # Plot standard deviation of density fluctuations.
PLOTBIAS    = 0    # Plot bias parameter.
PLOTSTDNDM  = 0    # Plot Sheth-Tormen mass function.
PLOTPSINT   = 0    # Plot integrand of contributions to non-linear power spectrum.
PLOTPS      = 0    # Plot non-linear power spectrum.

##### Variables for DLPS #####
n       =  1     # Power of Power Law.
a       =  1     # Scale Factor.
DHP     =  0.710 # Dimensionless Hubble Parameter.
H0      = 71.0   # (km/s/Mpc) Hubble constant. 
OMEGAM0 =  0.30  # Mass Density Parameter evaluated now.
OMEGAL0 =  0.70  # Cosmological Density Parameter evaluated now.
OMEGAB0 =  0.05  # Baryon Density Parameter evaluated now.
C       =  2.99792458e5   # (km/s) Speed of light. 
NORM    = (2 * np.pi)**3  # Normalization factor for dimensionless power spectrum
# Shape Parameter characterizes the dependence on cosmological parameters.
GAMMA   = OMEGAM0*DHP / (np.exp(OMEGAB0*(1 + 1.3/OMEGAM0)))
kLIN    = np.logspace(-2, 2, POINTS)  # (1/Mpc) Wavenumber  
# Numerical Parameters from simulations
ai      = np.array((2.34 , 3.89, 16.1 , 5.46, 6.71))
bi      = np.array((0.785, 0.05,  0.95, 0.169))    

##### Variables for SD #####
KMAX   = np.inf   # Maximum value of k to integrate to.
INTLIM = 500      # Upper bound on the number of subintervals used for sp.integrate.quad.
MSOLAR = 1.989e30 # (kg) Mass of Sun
RHO0   = 2.7e-27  # (kg/m**3) Density of universe. Ryden pg 11.
P1     = 3 / (4 * np.pi * RHO0) # constant in Top Hat Window function.
P2     = 4*np.pi  # constant in integrand.
KCONV  = DHP / (3.086e22) # conversion of k from h/Mpc to 1/m.
M_Mag  = np.logspace(12,21,POINTS) # (M/M_0) Mass in solar masses.
kVAR   = np.logspace(-6,2,POINTS) # (1/Mpc) Wavenumber

##### Variables for Bias Param #####
DELC = 1.68

##### Variables for Nonlinear Power Spectrum #####
MCO   =  MSOLAR*10**14 # Mass cutoff
# Intermediate calculation for the scale radius
SRCON = (3 / (800*np.pi*RHO0))**(1/3)    
DX    = np.sqrt(2.2e-16)           

# Parameters for Sheth-Tormen mass function.
ALPHA = 0.707
Q     = 0.3
A     = 0.322 




##### DIMENSIONLESS LINEAR POWER SPECTRUM #####
'''
This section of the project covers pages 42-45, sections 5.1-5.2, and reproduces 
Figure 5-1 for the Dimensionless Linear Power Spectrum.
'''

def MDP(af):
    '''
    calculate mass density parameter at given scale factor a.
    '''
    return OMEGAM0 / (OMEGAM0 + af**3 * OMEGAL0)

def cDP(af):
    '''
    calculate cosmological density parameter at given scale factor a.
    '''
    return 1 - MDP(af)

def g(af):
    '''
    calculate g, Relative Growth Factor, at given scale factor a.
    '''
    return 2.5 * MDP(af) / (MDP(af)**(4/7) - cDP(af) + (1 + MDP(af)/2) * \
           (1 + cDP(af) / 70))

def D(af):
    '''
    calcualte D, Linear Growth Factor, at given scale factor a. D_0 is the LGF. 
    (evaluated now at af=1)
    '''
    return af * g(af)

def q(kf):
    '''
    calculate q, Shape Parameter, for given range of wavenumbers k.
    '''
    return kf / (GAMMA * DHP)

def delH(nf):
    '''
    Not sure what this is called.
    '''
    return (1.94 * 10**(-5)) * OMEGAM0**(-bi[0] - bi[1] * np.log(OMEGAM0)) \
           * np.exp(-bi[2] * (nf - 1) - bi[3] * (nf - 1)**2)

def Af(nf):
    '''
    Not sure what this is called. Have a note in my copy of dissertation saying 
    it is the Normalization Factor.
    '''
    return delH(nf)**2 * ((C / H0)**(nf + 3)) / (4 * np.pi)

def LPS(kf, af, nf):
    '''
    Function for Linear Power Spectrum P(k, a) = P(k).
    (evaluated now at af=1)
    '''
    return Af(nf) * (kf**nf) * ((D(af) / D(1))**2) * (np.log(1 + ai[0] * q(kf)) \
           / (ai[0] * q(kf)))**2 / (1 + (ai[1] * q(kf)) +  ((ai[2] * q(kf))**2) \
           + ((ai[3] * q(kf))**3) + ((ai[4] * q(kf))**4))**0.5

if PLOTDLPS == 1:   
    '''
    This function differs from what is listed in Dr. Cooney's dissertation by 
    the NORM factor.
    '''
    # Dimensionless LPS
    graphLPS = NORM * (kLIN**3) * LPS(kLIN, a, n) / (2 * np.pi**2)
    
    # Figure parameters
    j+=1
    plt.figure(j, figsize=(FIGX,FIGY))
    plt.loglog(kLIN, graphLPS, '-')
    plt.title('Dimensionless Linear Power Spectrum')
    plt.xlabel('k (h/Mpc)')
    plt.ylabel(r'$\delta$ (k)')
    plt.savefig('DLPS.pdf')




##### STANDARD DEVIATION #####
'''
This portion of the project is devoted to numerically integrating the variance 
and plotting the standard deviation as a function of M in chapter 5, page 48. 
'''

def FTTHWF(kf, M):
    '''
    Fourier transform of top hat window function.
    '''
    return 3*(np.sin(kf * np.cbrt(M * P1)) - (kf * np.cbrt(M * P1)) \
              * np.cos(kf * np.cbrt(M * P1))) / (kf * np.cbrt(M * P1))**3

def VARIntegrand(kf, M):
    '''
    Integrand for variance of density contrast.
    '''
    return P2 * kf**2 * LPS(kf,a,n) * FTTHWF(KCONV*kf,M)**2

# Plot integrand for values of M to get an idea of when it behaves weirdly when 
# varying k to determine a proper interval for integration.

if PLOTVARINT == 1: 
    graphINTlow  = VARIntegrand(kVAR, MSOLAR * M_Mag[0])
    graphINTmed  = VARIntegrand(kVAR, MSOLAR * M_Mag[POINTS // 2])
    graphINThigh = VARIntegrand(kVAR, MSOLAR * M_Mag[-1])

    j+=1
    plt.figure(j, figsize=(FIGX,FIGY))
    plt.loglog(kVAR, graphINTlow)
    plt.loglog(kVAR, graphINTmed)
    plt.loglog(kVAR, graphINThigh)
    plt.title(r'Variance Integrand Low/Med/High $M/M_0$')
    plt.xlabel('k (h/Mpc)')
    plt.ylabel(r'Int(k)')
    plt.savefig('VARIntALL.pdf')
    
# Plot standard deviation.
   
def SDInt(M):
    '''
    Numerical integration for variance of density contrast. Returns graph of
    STD. scipy.integrate.quad returns tuple with the value of the integral and 
    the estimated value of error. The error for sp.integrate.quad on VARInteg over the 
    whole range of M/M0 is on the order of e-8 which is fairly accurate. I 
    originally thought this method would have a lot more error since the integrand
    is not well approximated as a polynomial and has many repeated cusps, but 
    it seems to work just fine.
    '''
    return np.sqrt(sp.integrate.quad(VARIntegrand, a=0, b=KMAX, args=(M), limit=INTLIM)[0])

if PLOTSD == 1:
    graphSD = np.zeros(POINTS)
    '''
    I have to use the property called broadcasting for np.array to avoid using
    loops at all. Otherwise the code is not as efficient. I came across a method
    in NumPy that vectorizes a function and believe that is what I need to do.
    '''
    for i in range(POINTS):
        graphSD[i] = SDInt(MSOLAR*M_Mag[i])
    
    j+=1
    plt.figure(j,figsize=(FIGX,FIGY))
    plt.loglog(M_Mag, graphSD)
    plt.title('Standard Deviation of Density contrast')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'$\sigma(M)$')
    plt.savefig('SD.pdf')
    
    # Needed later on in Sheth-Tormen mass function.
    j+=1
    plt.figure(j,figsize=(FIGX,FIGY))
    plt.semilogx(M_Mag, np.log(graphSD**(-1)))
    plt.title('Natural log of inverse SD')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'$ln(\sigma^{-1})$')
    plt.savefig('lnSDInv.pdf')
    
    
    

##### BIAS PARAMETER #####
# The bias parameter in the 2H contribution to the nonlinear power spectrum
# accounts for the difference in clustering between matter and dark matter
# density fields. 
    
def nu(M):
    '''
    Peak height in Jing (1998)
    '''
    return DELC / SDInt(M)

def bParam(M):
    '''
    Bias parameter used in the power spectrum of the 2-point correlation function.
    '''
    return (1 + (nu(M)**2 - 1)/DELC) * (1/(2*nu(M)**4) + 1)**(0.06-0.02*n)

if PLOTBIAS == 1:
    graphBIAS = np.zeros(POINTS)
    for i in range(POINTS):
        graphBIAS[i] = bParam(MSOLAR*M_Mag[i])
    
    j+=1
    plt.figure(j,figsize=(FIGX,FIGY))
    plt.loglog(M_Mag, graphBIAS)
    plt.title('Bias Parameter')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'$b(M)$')
    plt.savefig('BIAS.pdf')
    
    
    
    
##### NONLINEAR POWER SPECTRUM #####
    
def CP(M):
    '''
    Expression for the concentration parameter found in Jing and Suto.
    '''
    if M < MCO:
        return 5*SDInt(M)
    else:
        return 9*SDInt(M)
    
def SR(M, cp):
    '''
    Scale radius.
    '''
    return SRCON*M**(1/3) / cp

def delta(M, cp):
    '''
    Characteristic density.
    '''
    if M < MCO:
        # Density amplitude for NFW density profile
        return 200*cp**(3) / (3*(np.log(1 + cp) - cp/(1+cp))) 
    else:
        # Density amplitude for Moore density profile
        return 100*cp**(3) / np.log(1 + cp**(3/2)) 

def UP(q,M):
    '''
    Mixed density profile depending on value of M. MCO is the cutoff for 
    switching from one profile to the other.
    '''
    if M < MCO:
        '''
        Algebraic expression from Ma and Fry (2000) for fourier transform of NFW 
        density profile. 4% rms error
        '''
        return 4*np.pi*(np.log(np.exp(1)+1/q) - np.log(np.log(np.exp(1)\
                    +1/q)/3)) / ((1+q)**(1.1))**(2/1.1)
    else:
        '''
        Algebraic expression from Ma and Fry (2000) for fourier transform of Moore 
        density profile. 1% rms error
        '''
        return 4*np.pi*(np.log(np.exp(1) + 1/q) + 0.25*np.log(np.log(np.exp(1)\
                    + 1/q))) / (1 + 0.8*q**(1.5))
    
def lnInvSD(M):
    '''
    Intermediate caluculation in mass functions.
    '''
    return np.log(SDInt(M)**(-1))

def PSdndm(M):
    '''
    Press-Schecter mass/distribution function.
    '''
    dlnInvSD = sp.misc.derivative(lnInvSD,M,dx=M*DX,order=3)
    Nu       = nu(M)
    return np.sqrt(2/np.pi) * dlnInvSD * RHO0 * Nu * np.exp(-Nu**(2)/2) / M

def STdndm(M):
    '''
    Sheth-Tormen mass/distribution function.
    '''
    # Third order central finite difference method for estimating derivative
    # dlnInvSD = (lnInvSD(np.exp(np.log(M)+DX)) - lnInvSD(np.exp(np.log(M)-DX))) / 2*DX
    dlnInvSD = sp.misc.derivative(lnInvSD,M,dx=DX,order=3)
    NuP      = ALPHA**(-1/2)*nu(M)
    return (RHO0/M**2) * dlnInvSD * 2*A * (1 + 1/(NuP)**(2*Q)) * np.sqrt(NuP**(2)\
            / 2*np.pi) * np.exp(-NuP**(2)/2)
    
def PS1HInteg(M,kf):
    '''
    Integrand of 1 halo power spectrum contribution.
    '''
    cp = CP(M)
    sr = SR(M, cp)
    return STdndm(M) * (sr**(3) * delta(M, cp) * UP(KCONV*kf*sr,M))**2

def PS2HInteg(M,kf):
    '''
    Integrand of 1 halo power spectrum contribution.
    '''
    cp = CP(M)
    sr = SR(M, cp)
    return STdndm(M) * sr**3 * delta(M,cp) * UP(KCONV*kf*sr,M) * bParam(M)

if PLOTSTDNDM == 1:
    # Sheth-Tormen
    graphSTDNDM = np.zeros(POINTS)
    for i in range(POINTS):
        graphSTDNDM[i] = STdndm(MSOLAR*M_Mag[i])
        
    j+=1
    plt.figure(j, figsize=(FIGX,FIGY))
    plt.semilogx(M_Mag, graphSTDNDM)
    plt.title(r'Sheth-Tormen Mass Function')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'dn/dm')
    plt.savefig('STDNDM.pdf')
    
# This portion of code is not currently functioning because I am unable to
# get the integrand of the integrals to calculate properly using the current
# method for differentiation.

if PLOTPSINT == 1:
    '''
    Broadcasting of Numpy arrays is not magic so don't treat it like it is.
    '''
    PS1HINT = np.zeros((3,POINTS))
    PS2HINT = np.zeros((3,POINTS))
    
    for i in range(POINTS):
        PS1HINT[0,i] = PS1HInteg(MSOLAR*M_Mag[i], kLIN[0]          )
        PS1HINT[1,i] = PS1HInteg(MSOLAR*M_Mag[i], kLIN[POINTS // 2])
        PS1HINT[2,i] = PS1HInteg(MSOLAR*M_Mag[i], kLIN[POINTS - 1] )

        PS2HINT[0,i] = PS2HInteg(MSOLAR*M_Mag[i], kLIN[0]          )
        PS2HINT[1,i] = PS2HInteg(MSOLAR*M_Mag[i], kLIN[POINTS // 2])
        PS2HINT[2,i] = PS2HInteg(MSOLAR*M_Mag[i], kLIN[POINTS - 1] )
    
    j+=1
    plt.figure(j, figsize=(FIGX,FIGY))
    plt.loglog(M_Mag, PS1HINT[0])
    plt.loglog(M_Mag, PS1HINT[1])
    plt.loglog(M_Mag, PS1HINT[2])
    plt.title(r'PS1H Integrand Low/Med/High k')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'Int(M)')
    plt.savefig('PS1HIntALL.pdf')
    
    j+=1
    plt.figure(j, figsize=(FIGX,FIGY))
    plt.loglog(M_Mag, PS2HINT[0])
    plt.loglog(M_Mag, PS2HINT[1])
    plt.loglog(M_Mag, PS2HINT[2])
    plt.title(r'PS2H Integrand Low/Med/High $k$')
    plt.xlabel(r'$M/M_0$')
    plt.ylabel(r'Int(M)')
    plt.savefig('PS1HIntALL.pdf')

def PS1HInt(kf):
    return sp.integrate.quad(PS1HInteg, 0, 3, args=(kf,))[0]

if PLOTPS == 1:
    PS1HGRAPH = np.zeros(POINTS)
    for i in range(POINTS):
        PS1HGRAPH[i] = PS1HInt(kLIN[i])
    
    j+=1
    plt.figure(j,figsize=(FIGX,FIGY))
    plt.loglog(kLIN, PS1HGRAPH)
    plt.title('1H Power Spectrum Contribution')
    plt.xlabel('k (h/Mpc)')
    plt.ylabel(r'$P_{1H}(k)$')
    plt.savefig('P1H.pdf')




















