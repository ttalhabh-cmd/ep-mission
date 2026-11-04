"""Build a portable animation directly from sampled trajectory histories."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mission_engine import RE


def render_gif(track, baseline, title):
    ink, muted, teal, gold = "#e9f0fa","#a8bbcf","#5ae2c0","#ffba6a"
    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()
    small, body, header = font(14), font(17), font(25)
    h = track.altitude_km.to_numpy()
    t = track.time_days.to_numpy()
    lo, hi = min(h), max(h)
    if baseline is not None:
        lo,hi = min(lo,baseline.altitude_km.min()),max(hi,baseline.altitude_km.max())
    margin = max(0.2,(hi-lo)*0.09)
    lo,hi = lo-margin,hi+margin
    left,right,top,bottom = 385,864,118,325
    def points(times,altitudes):
        return [(left+(right-left)*day/t[-1],bottom-(bottom-top)*(height-lo)/(hi-lo)) for day,height in zip(times,altitudes)]
    series = points(t,h)
    frames = []
    for j in np.linspace(0,len(t)-1,65).astype(int):
        im = Image.new("RGB",(900,412),"#081526")
        d = ImageDraw.Draw(im)
        d.text((28,15),title[:53],fill=ink,font=header)
        d.text((29,55),f"Day {t[j]:.3f}    |    Altitude {h[j]:,.2f} km    |    Propellant used {track.propellant_used_kg.iloc[j]:.4g} kg",fill=teal,font=body)
        # Sampled phase is displayed directly. The orbital disk uses a fixed common scale.
        cx,cy = 160,220
        scale = 114/(RE/1000+max(h))
        earth = RE/1000*scale
        radius = (RE/1000+h[j])*scale
        d.ellipse((cx-earth,cy-earth,cx+earth,cy+earth),fill="#173959",outline="#34719a",width=2)
        d.text((cx,cy),"EARTH",font=body,fill=muted,anchor="mm")
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=teal,width=2)
        phi = track.phase_rad.iloc[j]
        x,y = cx+radius*np.cos(phi),cy-radius*np.sin(phi)
        d.ellipse((x-5,y-5,x+5,y+5),fill=ink)
        for frac in (0,.5,1):
            yy=bottom-(bottom-top)*frac
            d.line((left,yy,right,yy),fill="#294057")
            d.text((left-7,yy),f"{lo+(hi-lo)*frac:,.1f}",font=small,fill=muted,anchor="rm")
        d.text((left,89),"Mean altitude (km)",font=body,fill=ink)
        if baseline is not None:
            subset=baseline[baseline.time_days <= t[j]]
            p=points(subset.time_days,subset.altitude_km)
            if len(p)>1:d.line(p,fill=gold,width=3)
        if j:d.line(series[:j+1],fill=teal,width=3)
        px,py=series[j];d.ellipse((px-4,py-4,px+4,py+4),fill=teal)
        d.text((left,bottom+8),"0 d",font=small,fill=muted)
        d.text((right,bottom+8),f"{t[-1]:.3g} d",font=small,fill=muted,anchor="ra")
        d.text((28,366),"Orbital plane | sampled phase",font=small,fill=muted)
        d.text((left,366),"Submitted mission",font=small,fill=teal)
        if baseline is not None:d.text((left+170,366),"Unpowered comparison",font=small,fill=gold)
        frames.append(im)
    out=BytesIO()
    frames[0].save(out,format="GIF",save_all=True,append_images=frames[1:],duration=100,loop=0,disposal=2)
    return out.getvalue()
