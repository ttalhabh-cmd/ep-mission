from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from cluster import configure, allocate, from_record, QUICK
from transfer_engine import PRESETS, transfer, AU, EARTH_RADIUS
from mission_engine import G0, DAY


def movie(track,title,body):
    """Equatorial position and the history of osculating semi-major axis."""
    try: font=ImageFont.truetype("DejaVuSans.ttf",15)
    except OSError: font=ImageFont.load_default()
    t=track.time_days.to_numpy()
    earth=body=="Earth"
    unit='km' if earth else 'AU'
    semimajor=track.mean_radius_km.to_numpy()
    values=semimajor-EARTH_RADIUS/1000 if earth else semimajor/(AU/1000)
    orbit_scale=112/max(track.radius_km.max(),1.)
    lo,hi=values.min(),values.max()
    pad=max((hi-lo)*.08,1.0 if earth else .001)
    xx=360+480*t/max(t[-1],1e-16)
    yy=276-138*(values-lo+pad)/(hi-lo+2*pad)
    indices=np.unique(np.linspace(0,len(t)-1,65).astype(int))
    frames=[]
    for j in indices:
        im=Image.new("RGB",(900,365),"#172024");d=ImageDraw.Draw(im)
        def label(x,y,txt,color="#e8e7de"):
            d.text((x,y),txt,font=font,fill=color)
        label(24,16,title)
        label(24,45,f"Day {t[j]:,.2f}     Inclination {track.inclination_deg.iloc[j]:.2f} deg     Used {track.propellant_kg.iloc[j]:,.3g} kg","#edbc6b")
        cx,cy=155,193
        r=max(3,EARTH_RADIUS/1000*orbit_scale) if earth else 4
        d.ellipse((cx-r,cy-r,cx+r,cy+r),fill="#607477" if earth else "#e7ba69")
        x=cx+track.x_km.iloc[j]*orbit_scale;y=cy-track.y_km.iloc[j]*orbit_scale
        d.line((cx,cy,x,y),fill="#657875")
        d.ellipse((x-4,y-4,x+4,y+4),fill="#8adebc")
        # Display projected current orbit for circular solutions; an eccentric run uses only its actual position.
        if track.eccentricity.max()<1e-10:
            radius=track.radius_km.iloc[j]*orbit_scale
            ry=radius*abs(np.cos(np.deg2rad(track.inclination_deg.iloc[j])))
            d.ellipse((cx-radius,cy-ry,cx+radius,cy+ry),outline="#587f75")
        label(24,321,body+" / equatorial projection")
        label(360,90,("Mean orbital altitude" if earth else "Heliocentric semi-major axis")+f" ({unit})")
        for scale in (0,.5,1):
            gy=276-scale*138
            d.line((360,gy,840,gy),fill="#435152")
            label(846,gy-8,f"{lo-pad+scale*(hi-lo+2*pad):.3g}","#b6c2bd")
        if j:d.line(list(zip(xx[:j+1],yy[:j+1])),fill="#8adebc",width=3)
        label(360,305,f"0 d                                   {t[-1]:,.2f} d")
        frames.append(im)
    buffer=BytesIO()
    frames[0].save(buffer,format="GIF",save_all=True,append_images=frames[1:],duration=105,loop=0)
    return buffer.getvalue()


def six_way(df,cl,wet,fuel,days,duty,dv):
    table=[]
    for ident in QUICK:
        row=df.loc[ident]
        banks=[from_record(ident,row,b['installed'],b['reserve'],b['cant_deg'],b['formation']) for b in cl['banks']]
        run=allocate(banks,cl['available_w'])
        use=time=None
        status="Insufficient input power"
        if run['flow_kg_s']>0:
            use=wet*(-np.expm1(-dv/(run['isp_s']*G0)))
            time=use/run['flow_kg_s']/duty/DAY
            limits=[]
            if use>fuel:limits.append("propellant")
            if time>days:limits.append("time")
            status="Exceeds "+" and ".join(limits) if limits else "Within budgets"
        table.append(dict(Model=row['thruster_name'],Firing=sum(b['firing'] for b in run['banks']),
                **{"Input kW":run['power_w']/1000,"Axial mN":run['thrust_n']*1000,"Required kg":use,"Elapsed days":time,"Budget check":status}))
    return pd.DataFrame(table)


def transfer_page(df):
    st.caption("MISSION ANALYSIS   /   TRANSFERS")
    st.title("Transfer design")
    st.write("Size an engine cluster against the orbit change, available power and propellant budget.")
    preset=st.selectbox("Mission",list(PRESETS))
    body,start,end,i0,i1=PRESETS[preset];earth=body=="Earth";gto=preset=="GTO to GEO"
    method="Planar RK4" if gto else st.radio("Transfer method",["Edelbaum","Planar RK4"],horizontal=True)
    planar=method=="Planar RK4"
    st.caption("Cartesian RK4 · in-plane Lyapunov guidance" if planar else "Circular endpoints · common node · orbit-averaged steering")
    with st.container(border=True):
        st.subheader("Propulsion cluster")
        cl=configure(df,"trade")
    with st.form("transfer_inputs"):
        st.subheader("Boundary conditions")
        a,b=st.columns(2)
        unit="altitude (km)" if earth else "orbital radius (AU)"
        lower,upper=(120.,1000000.) if earth else (.1,5.)
        start=a.number_input(("Initial perigee " if gto else "Departure ")+unit,lower,upper,start,key="dep"+preset)
        end=b.number_input("Arrival "+unit,lower,upper,end,key="arr"+preset)
        apogee=a.number_input("Initial apogee altitude (km)",120.,1e6,35785.863) if gto else None
        i0=a.number_input("Departure inclination (deg)",0.,180.,i0,key="i0"+preset)
        i1=i0 if planar else b.number_input("Arrival inclination (deg)",0.,180.,i1,key="i1"+preset)
        solar_power=st.checkbox("Scale mean propulsion power with inverse-square solar flux",value=True) if planar and not earth else False
        st.subheader("Spacecraft and operating limits")
        a,b,c,d=st.columns(4)
        wet=a.number_input("Wet mass (kg)",1.,1000000.,2000.)
        fuel=b.number_input("Usable propellant (kg)",.001,999999.,600.)
        duty=c.number_input("Maximum duty (%)",.1,100.,90.)/100
        days=d.number_input("Time limit (days)",.01,3650.,900.)
        submit=st.form_submit_button("Calculate transfer",type="primary")
    with st.expander("Equations and assumptions"):
        st.latex(r"\Delta v=\sqrt{v_0^2+v_f^2-2v_0v_f\cos(\pi|\Delta i|/2)},\quad v=\sqrt{\mu/r}")
        st.write("Inclinations share a node and are converted to radians. The circular solver follows the Edelbaum velocity-space chord. Variable mass and constant mean axial thrust set its clock through the rocket equation. Edelbaum cases assume sustained input power. The planar solar option scales mean thrust and flow by min(1, (departure radius / current radius)²), equivalent to additional duty modulation with energy buffering. Lunar-distance and planet-radius cases end at an orbital radius. Encounter phasing, escape and capture require separate modelling.")
        st.write("Planar RK4 uses the uploaded Cartesian guidance law. GTO starts in an ellipse. Other cases start on a circle in the selected plane. Arrival tolerances are 1% in semi-major axis and eccentricity below 0.01. All transfer cases omit drag, eclipse and third bodies.")
    if submit:
        try:
            with st.spinner("Computing transfer…"):
                track,report=transfer(preset,start,end,i0,i1,wet,fuel,days,duty,cl,apogee,planar,solar_power)
                title=preset
                params=dict(mission=preset,method=method,inverse_square_solar=solar_power,body=body,start=start,end=end,endpoint_units=unit,i0_deg=i0,i1_deg=i1,
                            apogee_altitude_km=apogee,wet_kg=wet,fuel_kg=fuel,duty=duty,days=days,cluster=cl)
                comparison=None if planar else six_way(df.set_index("thruster_id"),cl,wet,fuel,days,duty,report['required_dv_m_s'])
                playback=movie(track,title,body)
                buf=BytesIO()
                with ZipFile(buf,'w',ZIP_DEFLATED) as z:
                    z.writestr("trajectory.csv",track.to_csv(index=False))
                    z.writestr("case.json",json.dumps(dict(inputs=params,result=report),indent=2))
                    z.writestr("trajectory.gif",playback)
                    if comparison is not None:z.writestr("six_system_comparison.csv",comparison.to_csv(index=False))
                st.session_state.transfer_output=(title,report,track,comparison,playback,buf.getvalue(),params)
        except ValueError as exc:st.error(str(exc))
    if "transfer_output" not in st.session_state:return
    title,report,track,comparison,playback,bundle,params=st.session_state.transfer_output
    last=track.iloc[-1]
    st.divider()
    st.subheader("Last calculated transfer")
    st.write(f"**{title}**  ·  {report['status']}  ·  {report['method']}")
    st.caption(" + ".join(f"{b['firing']} × {b['name']}" for b in params['cluster']['banks'])+f"  /  {params['wet_kg']:g} kg wet  /  {params['duty']:.0%} duty")
    if report.get('max_fractional_speed_per_rev',0)>.05:
        st.warning("The estimated speed change per orbit exceeds 5% along the planned arc. Treat the orbit-averaged result as a rough sizing estimate.")
    metrics=[("Delivered Δv",f"{last.delivered_dv_m_s/1000:.3f} km/s"),("Elapsed",f"{last.time_days:.2f} d"),("Used propellant",f"{last.propellant_kg:.3g} kg")]
    if comparison is not None:metrics.insert(0,("Edelbaum Δv",f"{report['required_dv_m_s']/1000:.3f} km/s"))
    for c,(k,v) in zip(st.columns(len(metrics)),metrics):c.metric(k,v)
    if comparison is not None:
        st.caption(f"Full transfer requirement: {report['required_propellant_kg']:.2f} kg, {report['required_firing_days']:.2f} firing days and {report['required_elapsed_days']:.2f} elapsed days.")
        with st.expander("Steering and transfer budgets"):
            st.write(f"Mean steering yaw from prograde: {report['initial_yaw_deg']:.2f}° at departure, {report['final_yaw_deg']:.2f}° at arrival. Positive yaw corresponds to increasing inclination in the averaged steering frame.")
            st.write(f"Propellant margin {params['fuel_kg']-report['required_propellant_kg']:+.2f} kg · Time margin {params['days']-report['required_elapsed_days']:+.2f} days.")
            st.caption(f"Coplanar, impulsive Hohmann reference: {report['coplanar_hohmann_dv_m_s']/1000:.3f} km/s for the radius change alone. The Edelbaum result also includes the selected inclination change.")
    st.image(playback,caption="Integrated or reconstructed states sampled across mission time. Intermediate revolutions may be skipped.")
    a,b=st.columns(2);a.download_button("Download trajectory GIF",playback,"trajectory.gif","image/gif")
    b.download_button("Download transfer package",bundle,"transfer.zip","application/zip")
    if comparison is not None:
        st.subheader("Six-system trade")
        st.caption("Each row replaces every bank with the named model. Installed counts, cold spares, cant, duty and the numerical power budget are held fixed.")
        st.dataframe(comparison,hide_index=True,width="stretch")
    else:
        st.line_chart(track,x="time_days",y="eccentricity",x_label="Days",y_label="Eccentricity")
