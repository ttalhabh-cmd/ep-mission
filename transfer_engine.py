"""Circular Edelbaum sizing and GTO circularisation with the supplied planar guidance."""
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from mission_engine import G0, DAY

AU = 149597870700.
# Input units are altitude km for Earth and orbital radius AU for Sun.
EARTH_RADIUS = 6378137.
MU_E = 3.986004418e14
MU_S = 1.32712440018e20
PRESETS = {
    "LEO to GEO": ("Earth",400.,35785.863,28.5,0.),
    "LEO to navigation orbit": ("Earth",500.,20200.,51.6,55.),
    "LEO to lunar distance": ("Earth",400.,384400.-6378.137,0.,0.),
    "Orbit lowering": ("Earth",1500.,500.,45.,45.),
    "Inclination change": ("Earth",800.,800.,45.,55.),
    "Earth to Mars orbital radius": ("Sun",1.,1.524,0.,0.),
    "Earth to Venus orbital radius": ("Sun",1.,0.723,0.,0.),
    "Custom Earth orbit": ("Earth",600.,2500.,10.,20.),
    "Custom heliocentric orbit": ("Sun",1.,2.,0.,0.),
    "GTO to GEO": ("Earth",250.,35785.863,0.,0.),
}

def size_circular(mu,r0,r1,i0,i1):
    if min(r0,r1,mu)<=0 or not (0<=i0<=180 and 0<=i1<=180) or abs(np.deg2rad(i1-i0)) >= 2:
        raise ValueError("Choose positive radii and an inclination change smaller than 114.59 degrees.")
    v0,v1=np.sqrt(mu/np.array([r0,r1]))
    angle=np.pi/2*np.deg2rad(i1-i0)
    dx=v1*np.cos(angle)-v0
    dy=v1*np.sin(angle)
    return v0,dx,dy,np.hypot(dx,dy)

def transfer(preset,start,end,i0,i1,wet,fuel,days,duty,cluster,apogee=None,cartesian=False,solar_power=False):
    body=PRESETS[preset][0]
    if not 0<fuel<wet or days<=0 or not 0<duty<=1:
        raise ValueError("Choose a positive duration, duty and propellant load below wet mass.")
    thrust=cluster['thrust_n']*duty
    flow=cluster['flow_kg_s']*duty
    if thrust<=0 or flow<=0:
        raise ValueError("The cluster has no firing units. Check power, cold spares and opposed-pair counts.")
    ve=thrust/flow
    earth=body=="Earth"
    mu=MU_E if earth else MU_S
    r0=EARTH_RADIUS+start*1000 if earth else start*AU
    r1=EARTH_RADIUS+end*1000 if earth else end*AU
    if earth and min(start,end)<120 or not earth and min(start,end)<0.1:
        raise ValueError("Earth endpoints must be at least 120 km high. Solar endpoints must be at least 0.1 AU.")
    if preset=="GTO to GEO" or cartesian:
        from planar_guidance import _run
        eccentric=preset=="GTO to GEO"
        ra=EARTH_RADIUS+apogee*1000 if eccentric and apogee is not None else r1 if eccentric else r0
        if eccentric and (not r0<ra or r1<r0):raise ValueError("The initial apogee must exceed perigee and the target must be at or above perigee.")
        if i0!=i1:raise ValueError("Planar propagation requires equal initial and final inclinations.")
        if thrust/(wet-fuel) > .2*mu/max(ra,r1)**2:raise ValueError("Acceleration exceeds this low-thrust guidance model's domain. Reduce duty or choose a smaller cluster.")
        v=np.sqrt(mu*(2/r0-2/(ra+r0)))
        s=np.array([r0,0.,0.,v,wet])
        eta = .3 if earth else .7
        exponent = 2 if solar_power and not earth else 0
        out,status=_run(s,mu,r1,1.,eta,thrust,flow,r0,exponent,days*DAY,wet-fuel,48.,6000,
                        EARTH_RADIUS+120e3 if earth else .05*AU)
        t,x,y,vx,vy,m,a=out.T
        h=x*vy-y*vx
        radius=np.hypot(x,y)
        ecc=np.hypot(vy*h/mu-x/radius,-vx*h/mu-y/radius)
        df=pd.DataFrame(dict(time_days=t/DAY,radius_km=radius/1000,mean_radius_km=a/1000,eccentricity=ecc,
                  inclination_deg=i0,propellant_kg=wet-m,delivered_dv_m_s=ve*np.log(wet/m),x_km=x/1000,
                  y_km=y*np.cos(np.deg2rad(i0))/1000,z_km=y*np.sin(np.deg2rad(i0))/1000))
        return df,dict(status=['Running','Target tolerance reached','Propellant exhausted','Time limit','Model boundary','Integration step limit'][status],
                       method='Planar Cartesian RK4 with Lyapunov guidance',body=body)
    v0,dx,dy,dv=size_circular(mu,r0,r1,i0,i1)
    if dv<1e-8:raise ValueError("The endpoint already matches the initial orbit. Change radius or inclination.")
    # Closest approach of the velocity chord gives the largest intermediate radius.
    closest=np.clip(-v0*dx/(dv*dv),0,1)
    max_r=mu/((v0+closest*dx)**2+(closest*dy)**2)
    required=wet*(-np.expm1(-dv/ve))
    fraction_per_orbit=thrust*2*np.pi*max_r**2/((wet-min(required,fuel))*mu)
    needed=required/flow
    stop=min(needed,fuel/flow,days*DAY)
    status="Target reached" if stop==needed else ("Propellant exhausted" if stop==fuel/flow else "Time limit")
    t=np.linspace(0,stop,501)
    used=t*flow
    delivered=-ve*np.log1p(-used/wet)
    u=np.clip(delivered/dv,0,1)
    vx=v0+u*dx
    vy=u*dy
    rad=mu/(vx**2+vy**2)
    inc=i0+np.rad2deg(2/np.pi*np.arctan2(vy,vx))
    phase=cumulative_trapezoid(np.sqrt(mu/rad**3),t,initial=0)
    df=pd.DataFrame(dict(time_days=t/DAY,radius_km=rad/1000,mean_radius_km=rad/1000,eccentricity=0.,
                         inclination_deg=inc,propellant_kg=used,delivered_dv_m_s=delivered,
                         x_km=rad*np.cos(phase)/1000,y_km=rad*np.sin(phase)*np.cos(np.deg2rad(inc))/1000,
                         z_km=rad*np.sin(phase)*np.sin(np.deg2rad(inc))/1000))
    # Yaw from prograde toward positive plane-change direction, in the averaged steering frame.
    angle=np.pi/2*np.deg2rad(i1-i0)
    yaw0=np.rad2deg(np.arctan2(dy,-dx))
    yawf=np.rad2deg(np.arctan2(-dx*np.sin(angle)+dy*np.cos(angle),-(dx*np.cos(angle)+dy*np.sin(angle))))
    a_transfer=(r0+r1)/2
    impulsive=(abs(np.sqrt(mu*(2/r0-1/a_transfer))-np.sqrt(mu/r0))
               +abs(np.sqrt(mu/r1)-np.sqrt(mu*(2/r1-1/a_transfer))))
    return df,dict(initial_yaw_deg=yaw0,final_yaw_deg=yawf,coplanar_hohmann_dv_m_s=impulsive,
                   method="Edelbaum circular endpoints",body=body,status=status,required_dv_m_s=dv,
                   required_propellant_kg=required,required_elapsed_days=needed/DAY,
                   max_fractional_speed_per_rev=fraction_per_orbit,required_firing_days=needed*duty/DAY,initial_accel_mm_s2=1000*thrust/wet)
