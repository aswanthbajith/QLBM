"""SI interpretation and nominal similarity parameters (not experimental validation)."""
from math import sqrt

def nominal(c):
    H,g,nu,sigma=(c[k] for k in ('H','gravity','nu_liquid','sigma'))
    return dict(Re=sqrt(g*H)*H/nu,Bo=(1-1.2/998)*g*H**2/sigma,
                characteristic_Mach=sqrt(3*g*H),t_star=c['steps']*sqrt(g/H),
                Cahn=c['xi']/H,tau_liquid=.5+3*nu)

def si_mapping(c,H_m=.1,g_si=9.81,rho_liquid=998.,rho_air=1.2,mu_water=.001,sigma_water=.072):
    if min(H_m,g_si,rho_liquid,rho_air,mu_water,sigma_water)<=0 or rho_air>=rho_liquid:raise ValueError('invalid physical parameters')
    dx=H_m/c['H'];dt=sqrt(c['gravity']*dx/g_si);nu_water=mu_water/rho_liquid
    u=sqrt(g_si*H_m);nu_lu=nu_water*dt/dx**2;sig_lu=sigma_water*dt**2/(rho_liquid*dx**3)
    return dict(H_m=H_m,dx_m=dx,dt_s=dt,duration_s=c['steps']*dt,
        modeled_nu_m2_s=c['nu_liquid']*dx**2/dt,modeled_sigma_N_m=c['sigma']*rho_liquid*dx**3/dt**2,
        target_nu_lu=nu_lu,target_sigma_lu=sig_lu,target_tau_liquid=.5+3*nu_lu,
        target_Re=u*H_m/nu_water,target_Bo=(rho_liquid-rho_air)*g_si*H_m**2/sigma_water,
        viscosity_excess=c['nu_liquid']/nu_lu,surface_tension_ratio=c['sigma']/sig_lu)
