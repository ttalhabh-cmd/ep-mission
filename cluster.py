"""Nominal-point cluster dispatch. Power is measured at the thruster inputs."""
from dataclasses import dataclass, asdict
from math import cos, sin, pi, radians, floor, isfinite
from mission_engine import G0

QUICK = ("EPT-0319","EPT-0349","EPT-0295","EPT-0198","EPT-0294","EPT-0292")

@dataclass(frozen=True)
class Bank:
    record: str
    name: str
    power_w: float
    thrust_mn: float
    isp_s: float
    installed: int
    reserve: int
    cant_deg: float
    formation: str = "Opposed pair"


def allocate(banks, available_w):
    if not isfinite(available_w) or available_w < 0:
        raise ValueError("Available power must be finite and nonnegative.")
    left = available_w
    force = flow = 0.
    dispatch = []
    for bank in banks:
        if (not 0 <= bank.reserve <= bank.installed or not 0 <= bank.cant_deg < 90
                or min(bank.power_w,bank.thrust_mn,bank.isp_s) <= 0):
            raise ValueError("Check the unit counts, cant and nominal operating point.")
        formations={"Parallel":1,"Opposed pair":2,"Triangular ring":3,"Square ring":4,"Hexagonal ring":6}
        if bank.formation not in formations or (bank.formation=="Parallel" and bank.cant_deg):
            raise ValueError("Parallel axial units require zero cant.")
        group = formations[bank.formation] if bank.cant_deg else 1
        possible = min(bank.installed-bank.reserve, floor((left+1e-9)/bank.power_w))
        active = possible // group * group
        draw = active*bank.power_w
        gross = active*bank.thrust_mn/1000
        axial = gross*cos(radians(bank.cant_deg))
        used = gross/(bank.isp_s*G0)
        flow += used
        force += axial
        left -= draw
        dispatch.append(dict(**asdict(bank), firing=active, draw_w=draw, axial_mn=axial*1000, flow_kg_s=used))
    return dict(banks=dispatch,thrust_n=force,flow_kg_s=flow,isp_s=force/(G0*flow) if flow else 0.,
                power_w=available_w-left,available_w=available_w)


def from_record(ident,row,count=1,reserve=0,cant=0.,formation="Opposed pair"):
    return Bank(ident,row['thruster_name'],float(row['P_nom_W']),float(row['T_nom_mN']),
                float(row['Isp_nom_s']),count,reserve,cant,formation)


def configure(df,key,first="EPT-0319"):
    import streamlit as st
    import pandas as pd
    lookup=df.set_index("thruster_id")
    count=st.selectbox("Thruster banks",list(range(1,7)),key=key+"_banks",
                       help="Banks share an input-power budget. Priority follows the bank order.")
    banks=[]
    for slot in range(count):
        prefix=f"{key}_{slot}_"
        pick,model=st.columns([1,2])
        families=["Six reference systems","All database records",*sorted(df.thruster_type.unique())]
        category=pick.selectbox(f"Bank {slot+1} · catalogue",families,index=1 if first not in QUICK else 0,key=prefix+"family")
        ids=(list(QUICK) if category==families[0] else
             lookup[lookup.thruster_type==category].index.tolist() if category!=families[1] else lookup.index.tolist())
        default=first if slot==0 else QUICK[slot%len(QUICK)]
        ident=model.selectbox(f"Bank {slot+1} · model",ids,index=ids.index(default) if default in ids else 0,
                   format_func=lambda i:f"{lookup.at[i,'thruster_name']} · {i}",key=prefix+"model")
        a,b,c,d=st.columns(4)
        n=a.number_input("Installed units",1,128,1,key=prefix+"units")
        reserve=b.number_input("Cold spares",0,n,0,key=prefix+"spares")
        formation=c.selectbox("Formation",["Parallel","Opposed pair","Triangular ring","Square ring","Hexagonal ring"],key=prefix+"formation")
        cant=d.number_input("Cant angle (deg)",0.,75.,0.,key=prefix+"cant",disabled=formation=="Parallel",
                            help="Canted rings fire complete groups, equally spaced in azimuth, to balance lateral thrust.")
        if formation=="Parallel":cant=0.
        banks.append(from_record(ident,lookup.loc[ident],n,reserve,cant,formation))
    requested=sum((b.installed-b.reserve)*b.power_w for b in banks)
    auto=st.checkbox("Supply full cluster power",True,key=key+"_auto")
    power=requested if auto else st.number_input("Available propulsion input (W)",0.,1e10,float(requested),key=key+"_bus")
    cl=allocate(banks,power)
    st.caption(f"Dispatch  /  {sum(b['firing'] for b in cl['banks'])} firing  ·  {cl['power_w']/1000:.3g} kW input  ·  {cl['thrust_n']*1000:.4g} mN axial  ·  {cl['isp_s']:.0f} s effective Isp")
    with st.expander("Bank allocation and formation"):
        # Diagram shows one repeating formation cell. Totals and spares appear in the dispatch table.
        diagrams=[]
        for idx,bank in enumerate(cl['banks']):
            n={"Parallel":1,"Opposed pair":2,"Triangular ring":3,"Square ring":4,"Hexagonal ring":6}[bank['formation']]
            d=[]
            for j in range(n):
                ang=2*pi*j/n-pi/2
                x=48+(23*cos(ang) if n>1 else 0)
                y=44+(23*sin(ang) if n>1 else 0)
                d.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="none" stroke="#86baa4" stroke-width="2"/>')
                if bank['cant_deg'] and n>1:
                    d.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+11*cos(ang):.1f}" y2="{y+11*sin(ang):.1f}" stroke="#d9b775"/>')
            diagrams.append(f'<svg width="100" height="98" viewBox="0 0 100 98" xmlns="http://www.w3.org/2000/svg"><circle cx="48" cy="44" r="38" fill="#152125" stroke="#445754"/>{"".join(d)}<text x="48" y="95" text-anchor="middle" fill="#c8d0c8" font-size="11">BANK {idx+1:02d}</text></svg>')
        st.html('<div style="display:flex;gap:16px;flex-wrap:wrap">' + ''.join(diagrams) + '</div>')
        st.caption("Rear view of one repeating formation cell per bank. Lines show the radial component of canted thrust; dimensions are schematic.")
        view=pd.DataFrame(cl['banks'])[['name','formation','installed','reserve','cant_deg','firing','draw_w','axial_mn']]
        st.dataframe(view,hide_index=True,width='stretch')
        st.caption("First bank has first call on power. Canted units fire in complete symmetric groups. All firing banks share the mission duty factor. Hardware fit, tank compatibility and PPU losses must be assessed separately.")
    return cl
