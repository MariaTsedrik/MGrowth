import numpy as np 
from scipy.integrate import odeint, quad
from scipy import interpolate
from scipy.special import hyp2f1


class MGrowth(object):
    def __init__(self, CosmoDict):
        """
        Minimal initialization requires the definition of a dictionary including
        only the cosmological parameters that are necessary to compute the expansion
        history and the scale factor(s).
        If the cosmology is not specified, hardcoded intial values are assumed.
        """

        if CosmoDict == None:
            self.omega0 = 0.3
            self.h = 0.68
            self.w0 = -1.
            self.wa = 0.
            self.a_arr = [1.]
        else:
            self.omega0 = CosmoDict['Omega_m']
            self.h = CosmoDict['h']
            self.w0 = CosmoDict['w0']
            self.wa = CosmoDict['wa']
            self.a_arr = CosmoDict['a_arr']
             
        self.a_start = 1.e-4
        self.aa = np.hstack(([self.a_start], self.a_arr))
        self.aa_interp = np.logspace(-5, 1.5, 512)

    def Omega_m(self, a, omega0=0.3, w0=-1, wa=0):
        """
        Computes a function :math:`\Omega_\mathrm{m}(a)` as a fuction of the scale-factor:
        
        .. math::
              \Omega_\mathcal{m}(a) = \\frac{\Omega_{m,0} a^{-3}}
              {E^2(a)} = \\frac{\Omega_{m,0} a^{-3}}
              {\Omega_{m,0} a^{-3}+\Omega_\mathrm{DE}(a)}
        assuming a flat universe with the expansion history :math:`H(a)=H_0 E(a)` the dark 
        energy component that evolves as 
        :math:`\Omega_\mathrm{DE}(a) = (1-\Omega_\mathrm{m, 0}) a^{-3(1+w_0+w_a)} \mathrm{e}^{3(a-1)w_a}`.

        Args:
            a       (array):  scale factor array, is strictly increasing

            omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            w0      (scalar): dark energy equation of state parameter

            wa      (scalar): dark energy equation of state parameter

        Returns:
            array: Omega_m(a) 
        """    

        omegaL = (1.-omega0) * a**(-3.*(1.+w0+wa)) * np.exp(3.*(-1.+a)*wa)
        E2 = omega0/a**3 + omegaL
        return omega0/a**3/E2

    def dlnH_dlna(self, a, omega0=0.3, w0=-1, wa=0):
        """
        Computes the derivative of the ln(expansion) with respect to ln(scale factor):

        .. math::
              \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} = 
               -\\frac{3}{2} \\frac{\Omega_\mathrm{m} a^{-3} + (1+w_0+w_a[1-a])\Omega_\mathrm{DE}(a)}
              {\Omega_{m,0} a^{-3}+\Omega_\mathrm{DE}(a)}

        Args:
            a       (array):  scale factor array, is strictly increasing

            omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            w0      (scalar): dark energy equation of state parameter

            wa      (scalar): dark energy equation of state parameter

        Returns:
            array: dlnH_dlna(a)
        """   

        omegaL = (1.-omega0) * a**(-3.*(1.+w0+wa)) * np.exp(3.*(-1.+a)*wa)
        E2 = omega0/a**3 + omegaL
        return  -1.5 * (omega0/a**3 + (1+w0+wa*(1.-a))*omegaL)/E2
    
    def Friction(self,  a, omega0=0.3, h=0.68, w0=-1, wa=0, xi=0):
        """
        Computes the modification to Euler equation in the interacting dark energy scenario.

        Args:
            a       (array):  scale factor array, is strictly increasing

            omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            h       (scalar): value of the Hubble constans as in :math:`H_0 = 100 h/` Mpc
            
            w0      (scalar): dark energy equation of state parameter

            wa      (scalar): dark energy equation of state parameter

            xi       (scalar): scattering strength

        Returns:
            array: Friction(a)
        """  

        c3 = 0.09747163203504212
        omegaL = (1.-omega0) * a**(-3.*(1.+w0+wa)) * np.exp(3.*(-1.+a)*wa)
        E = np.sqrt(omega0/a**3 + omegaL)
        Fr = (1 + w0 + wa*(1.-a))*xi*omegaL*h*c3/E
        return Fr

    def D_derivatives(self, D, a, omega0=0.3, w0=-1, wa=0):
        """
        Function that is used with scipy.odeint to solve the following differential 
        equation for the growth factor :math:`D(a)`:

        .. math::
            \ddot{D} +\left( 3 + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \Omega_\mathrm{m}(a) \\frac{D}{a^2} = 0

        with :math:`\dot{}` denoting a derivative with respect to the scale factor.

        Args:
            D       (array): growth factor, is strictly increasing

            a       (array):  scale factor array

            omega0  (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            w0      (scalar): dark energy equation of state parameter

            wa      (scalar): dark energy equation of state parameter


        Returns:
            array: D(a)
        """   

        return [D[1], -D[1]/a*(3.+self.dlnH_dlna(a, omega0, w0, wa))+1.5*D[0]/a**2*self.Omega_m(a, omega0, w0, wa)]
   
    def MG_D_derivatives(self, D, a, mu_interp, omega0=0.3, w0=-1, wa=0):
        """
        Function that is used with scipy.odeint to solve the following differential 
        equation for the modified growth factor :math:`D(a)`:

        .. math::
            \ddot{D} +\left( 3 + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \mu(a) \Omega_\mathrm{m}(a) \\frac{D}{a^2}= 0

        with :math:`\dot{}` denoting a derivative with respect to the scale factor and 
        :math:`\mu = \\frac{G_\mathrm{eff}(a)}{G}` being a modification to the gravitational constant.

        Args:
            D           (array): growth factor

            a           (array):  scale factor array, is strictly increasing

            mu_interp   (interpolator):  interpolator for the modification to the gravitational constant

            omega0      (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            w0          (scalar): dark energy equation of state parameter

            wa          (scalar): dark energy equation of state parameter


        Returns:
            array: D(a)
        """   

        return [D[1], -D[1]/a*(3.+self.dlnH_dlna(a, omega0, w0, wa))+1.5*mu_interp(a)*D[0]/a**2*self.Omega_m(a, omega0, w0, wa)]
   

    def IDE_D_derivatives(self, D, a, omega0=0.3, h=0.68, w0=-1, wa=0, xi=0):
        """
        Function that is used with scipy.odeint to solve the following differential 
        equation for the modified growth factor :math:`D(a)` in iteracting dark energy:

        .. math::
            \ddot{D} +\left( 3 + \mathrm{Friction} + \\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}} \\right) \\frac{\dot{D}}{a}-\\frac{3}{2} \Omega_\mathrm{m}(a) \\frac{D}{a^2} = 0

        with :math:`\dot{}` denoting a derivative with respect to the scale factor and 
        :math:`\mathrm{Friction} = (1+w(a))\\xi \\frac{\Omega_\mathrm{DE}(a) \\rho_\mathrm{crit}}{H(a)}` being a modification to the Euler equation.

        Args:
            D           (array): growth factor

            a           (array):  scale factor array, is strictly increasing

            omega0      (scalar): value of :math:`\Omega_\mathrm{m}(a=1) = \Omega_\mathrm{m,0}`

            w0          (scalar): dark energy equation of state parameter

            wa          (scalar): dark energy equation of state parameter

            xi          (scalar): scattering strength


        Returns:
            array: D(a)
        """   

        return [D[1], -D[1]/a*(3.+ self.Friction(a, omega0, h, w0, wa, xi) + self.dlnH_dlna(a, omega0, w0, wa))+1.5*D[0]/a**2*self.Omega_m(a, omega0, w0, wa)]

class LCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
 
    def growth_parameters(self):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the :math:`\Lambda` CDM cosmology.

        Returns:
            array: D(a), f(a)
        """   

        D, dDda = odeint(self.D_derivatives, [self.a_start, 1.], self.aa, args=(self.omega0, -1., 0.)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:]


class wCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
  
    def growth_parameters(self):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the wCDM cosmology with :math:`w=w_0` specialised by the initalisation.

        Returns:
            array: D(a), f(a)
        """   

        D, dDda = odeint(self.D_derivatives, [self.a_start, 1.], self.aa, args=(self.omega0, self.w0, 0.)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:]

class w0waCDM(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
   
    def growth_parameters(self):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in the :math:`w_0 w_a` CDM cosmology.

        Returns:
            array: D(a), f(a)
        """   

        D, dDda = odeint(self.D_derivatives, [self.a_start, 1.], self.aa, args=(self.omega0, self.w0, self.wa)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:] 
    
class IDE(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
   
    def growth_parameters(self, xi=0.):
        """
        Computes growth factor :math:`D` and growth rate :math:`f`
        at the scale factor specified by initialisation in the :math:`w_0 w_a A` CDM cosmology, also
        known as interacting dark energy. In this abbreviation :math:`w_0 w_a` are the parameters from the 
        equation of state for dark energy: :math:`w(a) = w_0 + w_a (1-a)` and parameter :math:`A = (1+w(a)) \\xi` 
        is introduced in such way to allow us to sample the parameters space with clear definition of the 
        :math:`w(a) \sim -1` case.

        Args:
        xi       (scalar): scattering strength
        
        Returns:
            array: D(a), f(a)
        """   

        assert xi>=0
        D, dDda = odeint(self.IDE_D_derivatives, [self.a_start, 1.], self.aa, args=(self.omega0, self.h, self.w0, self.wa, xi)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:]     
    

class fR_HS(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def growth_parameters(self, k_arr, fR0=1e-9):
        """
        Computes scale-dependent growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation in a f(R) gravity in the :math:`n=1` Hu-Sawicki model 
        :math:`f(R) = -m^2 \\frac{c_1 (R/m^2)^n}{c_2(R/m^2)+1}` with :math:`\mu(a, k) = 1 + \\frac{1}{3} \\frac{k^2}{k^2 + \\frac{a^2}{3 k^2}}`
        and :math:`f_{RR}(a) = \\frac{n(n+1)}{m^2} \\frac{c_1}{c_2^2} \left( \\frac{m^2}{R}  \\right)^{n+2}`
        where :math:`\\frac{c_1}{c_2^2} = -\\frac{1}{n} \left( 1 + \\frac{4 \Omega_\mathrm{DE,0}}{\Omega_\mathrm{m,0}} \\right)^{n+1} f_{R0}`,
        :math:`R(a) = m^2 \left( \\frac{3}{a^3} + 2 \\frac{c_1}{c_2} \\right)`, :math:`\\frac{c_1}{c_2} = 6 \\frac{\Omega_\mathrm{DE,0}}{\Omega_\mathrm{m,0}}`
        and :math:`m^2 = H_0^2 \Omega_\mathrm{m,0}`.
        For :math:`\Lambda` CDM scenario :math:`f_{R0} = 0`, typical scenarios include a weak deviation
        with :math:`f_{R0} = -10^{-6}` and a stronger deviation with :math:`f_{R0} = -10^{-5}`.
        Only :math:`\Lambda` CDM expansion is allowed due to the model specifics.

        Args:
            fR0       (positive float):  absolute value of the modification at :math:`a=1`, a positive number between :math:`10^{-9}` and :math:`10^{-2}`

            k         (array):   scales in :math:`h/` Mpc

        Returns:
            array: D(k, a), f(k, a)
        """ 

        assert fR0 > 0.
        c0 = 1./2997.92458 # H = h/Mpc c0
        var1 = [(k_i/self.aa_interp)**2  for k_i in k_arr]
        mu_fR = np.array([1. + 1./3* var1_i/(var1_i+ ( self.omega0/self.aa_interp**3 - 4.*(self.omega0 -1.) )**3/( 2. * (4. - 3.*self.omega0)**2 * fR0 / c0**2)) for var1_i in var1])
        mu_fR_interp = [interpolate.interp1d(self.aa_interp, mu_fR[i, :]) for i in range(len(k_arr))]
        D_f_i = [odeint(self.MG_D_derivatives, [self.a_start, 1.], self.aa, args=(mu_fR_interp_i, self.omega0, -1., 0.)).T for  mu_fR_interp_i in mu_fR_interp]
        D = np.array([D_i for D_i, _ in D_f_i])
        f = np.array([self.aa * dDda_i / D_i for D_i, dDda_i in D_f_i])

        return D[:,1:], f[:, 1:]    
    
class nDGP(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)

    def beta(self, a, omegarc):
        """
        Computes :math:`\\beta(a) = 1 + \\frac{E(a)}{\sqrt{\Omega_\mathrm{rc}}} \left( 1 + \\frac{1}{3}\\frac{\mathrm{d} \ln{H}}{\mathrm{d} \ln{a}}  \\right)`
        where :math:`\Omega_\mathrm{rc} = \\frac{1}{4 H_0^2 r_c^2}`.

        Args:
            a         (array):   values of scalar factors

            omegarc   (float):  value of the modification, a positive number between 1.e-6 and 1.e6 
            
        Returns:
            array: beta(a)
        """  

        omegaL = (1.-self.omega0) * a**(-3.*(1.+self.w0+self.wa)) * np.exp(3.*(-1.+a)*self.wa)
        E = np.sqrt(self.omega0/a**3 + omegaL)
        return 1. + E/np.sqrt(omegarc)  * (1. + 1./3 * self.dlnH_dlna(a, self.omega0, self.w0, self.wa))

    def growth_parameters(self, omegarc=1.e-6):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation for a nDGP cosmology with :math:`\mu(a) = 1 + \\frac{1}{3\\beta(a)}`.
        For :math:`\Lambda` CDM scenario :math:`\Omega_\mathrm{rc} = 0`, typical scenarios include a weak deviation
        with :math:`\Omega_\mathrm{rc} = 0.01` for :math:`r_c = 5 H_0^{-1}` and a stronger deviation with 
        :math:`\Omega_\mathrm{rc} = 0.25` for :math:`r_c = H_0^{-1}`.

        Args:
            omegarc   (float):  value of the modification, a positive number between 1.e-6 and 1.e6 

        Returns:
            array: D(a), f(a)
        """   
        
        mu_nDGP = 1. + 1./(3.*self.beta(self.aa_interp, omegarc))
        mu_nDGP_interp = interpolate.interp1d(self.aa_interp, mu_nDGP)
        D, dDda = odeint(self.MG_D_derivatives, [self.a_start, 1.], self.aa, args=(mu_nDGP_interp, self.omega0, self.w0, self.wa)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:] 

class Linder_gamma(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
   
    def growth_parameters(self, gamma=0.55):
        """
        Computes growth rate :math:`f = \Omega_\mathrm{m}(a)^\gamma` and 
        :math:`D(a)=D_\mathrm{ini} \exp{\int_{a_\mathrm{ini}}^a \mathrm{d}\\tilde{a} f(\\tilde{a})/\\tilde{a} }`
        with :math:`D_\mathrm{ini} = D_{\Lambda \mathrm{CDM}} (a=10^{-4})`
        at the scale factor specified by initialisation.

        Args:
            gamma (float): growth index, equals 0.555 in standard cosmology

        Returns:
            array: D(a), f(a)
        """ 

        f = self.Omega_m(self.aa[1:])**gamma
        func = lambda a_: self.Omega_m(a_)**gamma/a_
        Dini = self.a_start * hyp2f1(1./3, 1., 11./6, (self.omega0 - 1.) / self.omega0 * self.a_start**3)
        D = [(Dini * np.exp(quad(func, self.a_start, a_i)[0])) for a_i in self.aa[1:]]
        return D, f      
    
class Linder_gamma_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
   
    def growth_parameters(self, gamma0=0.55, gamma1=0.):
        """
        Computes growth rate :math:`f = \Omega_\mathrm{m}(a)^{\gamma(a)}` and 
        :math:`D(a)=D_\mathrm{ini} \exp{\int_{a_\mathrm{ini}}^a \mathrm{d}\\tilde{a} f(\\tilde{a})/\\tilde{a} }`
        with :math:`D_\mathrm{ini} = D_{\Lambda \mathrm{CDM}} (a=10^{-4})`
        at the scale factor specified by initialisation. The time parameterisation of 
        the growth index is taken from arXiv: 2304.07281, 
        namely :math:`\gamma(a) = \gamma_0 + \gamma_1 \\frac{(1-a)^2}{a}`.

        Args:
            gamma0 (float): growth index, equals 0.555 in standard cosmology
            gamma1 (float): growth index time component, equals 0 in standard cosmology 

        Returns:
            array: D(a), f(a)
        """ 

        f = self.Omega_m(self.aa[1:])**(gamma0 + gamma1 * (1.-self.aa[1:])**2/self.aa[1:])
        func = lambda a_: self.Omega_m(a_)**(gamma0 + gamma1 * (1.-a_)**2/a_)/a_
        Dini = self.a_start * hyp2f1(1./3, 1., 11./6, (self.omega0 - 1.) / self.omega0 * self.a_start**3)
        D = [Dini * np.exp(quad(func, self.a_start, a_i)[0]) for a_i in self.aa[1:]]
        return D, f              
    
class mu_a(MGrowth):
    def __init__(self, CosmoDict=None):
        super().__init__(CosmoDict)
   
    def growth_parameters(self, mu_interp):
        """
        Computes growth factor :math:`D` and growth rate :math:`f = \\frac{\mathrm{d} \ln{D}}{\mathrm{d} \ln{a}}`
        at the scale factor specified by initialisation for a modified cosmology with customed :math:`\mu(a)`.

        Args:
            mu_interp   (interpolator):  interpolator for the modification to the gravitational constant, should 
            allow for :math:`10^{-5} \geq a \geq 1.5` or :math:`10^{-5} \geq a \geq a_{max}+0.5`

        Returns:
            array: D(a), f(a)
        """ 
        D, dDda = odeint(self.MG_D_derivatives, [self.a_start, 1.], self.aa, args=(mu_interp, self.omega0, self.w0, self.wa)).T  
        f = self.aa*dDda/D
        return D[1:], f[1:]      
