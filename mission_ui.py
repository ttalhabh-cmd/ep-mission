from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
import streamlit as st
from mission_engine import Mission, MODES, run_mission
from mission_playback import render_gif
from cluster import configure

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "reference_media"
LABELS = ["VLEO and ABEP", "GEPC", "IAC", "Fusion", "Vacuum chamber", "Ariane", "CubeSat", "VLEO orbit viewers", "Presentation clips", "Other lecture media", "Uploaded · orbit transfers", "Uploaded · VLEO digital twin", "Uploaded · chamber particle tracing"]


def reference_library(suggested=None):
    records = json.loads((ASSETS/"index.json").read_text())
    groups = sorted({r['group'] for r in records} | {r['upload_group'] for r in records if 'upload_group' in r})
    group = st.selectbox("Collection",groups,index=groups.index(suggested) if suggested in groups else 0,
                         format_func=lambda g:LABELS[int(g[:2])-1])
    candidates=[r for r in records if r['group']==group or r.get('upload_group')==group]
    row = st.selectbox("Reference animation",candidates,format_func=lambda r:f"{r['file'][:3]} · {r['title']}")
    path=ASSETS/row['file']
    st.caption(f"Original asset · {row['origin']}")
    if path.suffix==".mp4": st.video(str(path))
    elif path.suffix==".gif": st.image(str(path))
    else: st.info("Download this interactive viewer or MATLAB source and run it locally.")
    st.download_button("Download reference asset",path.read_bytes(),file_name=Path(row['original']).name)
    if 'source_bundle' in row:
        st.caption("Uploaded project · " + row['upload_name'])
        source_path=ASSETS/row['source_bundle']
        st.download_button("Download project scripts, data and final frames",source_path.read_bytes(),source_path.name,"application/zip")
    st.caption("Renderer / provenance · "+row['source'])
    st.link_button("Full archive with original scripts","https://chatgpt.com/library?file_id=libfile_0df72c2c52148191bff22d16e58a9059&account_id=personal")


def design(df):
    st.caption("MISSION ANALYSIS   /   EARTH OPERATIONS")
    st.title("Earth orbit operations")
    case_type=st.radio("Mission type",MODES,horizontal=True)
    cl=None
    if case_type!=MODES[2]:
        with st.container(border=True):
            st.subheader("Propulsion cluster")
            cl=configure(df,"earth","EPT-0313")
    ideal=case_type==MODES[3]
    with st.form("mission_inputs"):
        left,right=st.columns(2)
        with left:
            st.subheader("Orbit")
            start=st.number_input("Initial altitude (km)",121.,42000. if ideal else 500.,400. if ideal else 250.,key=f"start_{ideal}")
            target=start
            if case_type in (MODES[0],MODES[3]):
                target=st.number_input("Target altitude (km)",121.,42000. if ideal else 500.,35786. if ideal else 275.,key=f"target_{ideal}")
            days=st.number_input("Maximum mission time (days)",0.01,3650.,365. if ideal else 15.,key=f"days_{ideal}")
            solar=st.selectbox("Atmosphere case · F10.7 (sfu)",[70,150,220],index=1,disabled=ideal)
        with right:
            st.subheader("Spacecraft")
            mass=st.number_input("Initial wet mass (kg)",1.,100000.,500. if ideal else 150.,key=f"mass_{ideal}")
            fuel=st.number_input("Usable propellant in wet mass (kg)",0.,99999.,100. if ideal else 8.,key=f"fuel_{ideal}")
            area=st.number_input("Frontal area (m²)",0.001,1000.,1.,format="%.3f",disabled=ideal)
            cd=st.number_input("Drag coefficient",0.1,5.,2.2,disabled=ideal)
        a,b=st.columns(2)
        duty=a.slider("Maximum on-time (%)",0,100,80,disabled=case_type==MODES[2])
        submitted=st.form_submit_button("Run propagation",type="primary")
    with st.expander("Propagation model and limits"):
        st.write("The backend integrates circular mean altitude, orbital phase and remaining propellant. VLEO cases use the archived NRLMSISE-00 density table with a fixed solar case. The ideal circular transfer uses a spherical Earth in vacuum. All orbits are coplanar.")
        st.write("Thrust and specific impulse use the selected nominal point. Available power below the nominal requirement prevents firing. Duty limits mean thrust. Station keeping commands the mean thrust needed to balance drag, capped by available thrust. Treat pulsed devices as an averaged nominal point. Eclipse, attitude, transient throttle behaviour and third-body gravity are outside this model.")
        st.caption("Propagation ends at the time limit, target altitude, propellant exhaustion or the 120 km cutoff, whichever applies. Early cutoff is a model boundary. The unpowered comparison runs to the submitted mission’s end time or its own cutoff.")
    if submitted:
        powered=bool(cl and cl['thrust_n']>0)
        args=Mission(case_type,start,target,days,mass,fuel,area,cd,solar,
                     cl['thrust_n']*1000 if powered else 1.,cl['isp_s'] if powered else 1.,
                     duty/100 if powered else 0.,cl['power_w'] if powered else 1.,cl['available_w'] if cl else 0.)
        try:
            with st.spinner("Propagating orbit and rendering mission playback…"):
                track,end,coast,coast_end=run_mission(args)
                label=" + ".join(f"{bank['firing']} × {bank['name']}" for bank in cl['banks']) if cl else ""
                title=case_type+" | "+label if cl else case_type
                movie=render_gif(track,coast,title)
                summary=dict(mission=asdict(args),cluster=cl,
                             termination=end,coast_termination=coast_end)
                out=BytesIO()
                with ZipFile(out,'w',ZIP_DEFLATED) as z:
                    z.writestr("inputs_and_status.json",json.dumps(summary,indent=2))
                    z.writestr("mission.csv",track.to_csv(index=False))
                    if coast is not None:z.writestr("unpowered.csv",coast.to_csv(index=False))
                    z.writestr("mission.gif",movie)
                st.session_state.mission_result=(title,track,end,coast,coast_end,movie,out.getvalue(),summary)
        except ValueError as exc:
            st.error(str(exc))
    if "mission_result" not in st.session_state:return
    title,track,end,coast,coast_end,movie,bundle,summary=st.session_state.mission_result
    st.divider()
    st.subheader("Last completed run")
    args=summary['mission']
    st.write(f"**{title}**  ·  Start {args['start_km']:g} km  ·  {args['wet_kg']:g} kg wet  ·  {end}")
    if args['mode']!=MODES[2] and (args['available_w']<args['nominal_w'] or args['duty']==0 or args['fuel_kg']==0):
        st.info("No thrust was available under the submitted power, duty and propellant limits.")
    last=track.iloc[-1]
    for col,(label,value) in zip(st.columns(4),[("Elapsed",f"{last.time_days:.3g} d"),("Final altitude",f"{last.altitude_km:,.2f} km"),("Propellant used",f"{last.propellant_used_kg:.4g} kg"),("Delivered Δv",f"{last.delivered_dv_m_s:.3g} m/s")]):
        col.metric(label,value)
    st.image(movie,caption="Orbit-plane position sampled at equal mission-time intervals. Long runs skip revolutions between frames.")
    aa,bb=st.columns(2)
    aa.download_button("Download mission animation",movie,"mission.gif","image/gif")
    bb.download_button("Download full run",bundle,"mission_run.zip","application/zip")
    if coast is not None:
        st.caption(f"Unpowered comparison ended at {coast.altitude_km.iloc[-1]:,.2f} km after {coast.time_days.iloc[-1]:.3g} days ({coast_end.lower()}).")
    st.line_chart(track,x="time_days",y=["thrust_mN","drag_mN"],x_label="Elapsed time (days)",y_label="Force (mN)")
    ref = {"VLEO reboost":"04_Power_limited_reboost","VLEO station keeping":"02_Altitude_keeping","Passive decay":"01_Atmosphere_and_drag","Ideal circular transfer":"05_Orbit_transfer_response"}
    with st.expander("Related archived animation"):
        records=json.loads((ASSETS/"index.json").read_text())
        item=next(r for r in records if Path(r['original']).stem==ref[args['mode']])
        st.caption("Fixed reference case · "+item['origin'])
        st.video(str(ASSETS/item['file']))
