"""Orbit-averaged Earth mission screening. State uses SI units except altitude (km)."""
from dataclasses import dataclass, replace
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

DATA = Path(__file__).resolve().parent / "data"
CONSTANTS = json.loads((DATA / "parameters.json").read_text())["constants"]
MU, RE, G0 = (CONSTANTS[k] for k in ("mu_m3_s2", "earth_radius_m", "g0_m_s2"))
DAY = 86400
MODES = ("VLEO reboost", "VLEO station keeping", "Passive decay", "Ideal circular transfer")


@dataclass(frozen=True)
class Mission:
    mode: str
    start_km: float
    target_km: float
    days: float
    wet_kg: float
    fuel_kg: float
    area_m2: float
    cd: float
    solar: int
    thrust_mn: float
    isp_s: float
    duty: float
    nominal_w: float
    available_w: float

    def validate(self):
        values = [v for v in vars(self).values() if not isinstance(v, str)]
        if not np.isfinite(values).all() or self.mode not in MODES:
            raise ValueError("Choose a valid mission and finite numeric inputs.")
        if not (0 <= self.fuel_kg < self.wet_kg and 0 < self.days <= 3650 and 0 <= self.duty <= 1):
            raise ValueError("Check duration, duty and mass. Propellant must be less than wet mass.")
        if min(self.area_m2, self.cd, self.thrust_mn, self.isp_s, self.nominal_w) <= 0 or self.available_w < 0:
            raise ValueError("Geometry and nominal operating-point values must be positive.")
        upper = 42000 if self.mode == MODES[3] else 500
        if self.solar not in (70,150,220) or not 120 < self.start_km <= upper:
            raise ValueError(f"Initial altitude must be above 120 km and at most {upper} km.")
        if self.mode in (MODES[0], MODES[3]) and not self.start_km < self.target_km <= upper:
            raise ValueError(f"Choose a target above the initial orbit and at most {upper} km.")


@lru_cache(maxsize=3)
def density_table(solar):
    df = pd.read_csv(DATA / "atmosphere_nrlmsise00.csv")
    subset = df[df.F107_sfu == solar]
    return subset.altitude_km.to_numpy(), subset.mass_density_kg_m3.to_numpy()


def propagate(case, coast=False):
    case.validate()
    vacuum = case.mode == MODES[3]
    passive = coast or case.mode == MODES[2]
    max_thrust = case.thrust_mn * 1e-3 * case.duty
    if passive or case.available_w < case.nominal_w or case.fuel_kg == 0:
        max_thrust = 0.0
    # Reject cases with large fractional changes in orbital speed per revolution.
    outer = case.target_km if case.mode in (MODES[0], MODES[3]) else case.start_km
    if max_thrust * 2*np.pi*(RE+outer*1000)**2 / ((case.wet_kg-case.fuel_kg)*MU) > 0.05:
        raise ValueError("Thrust is too large for this orbit-averaged approximation. Increase mass or reduce duty.")
    heights, densities = density_table(case.solar)

    def forces(h):
        radius = max(RE + 1000*h, RE)
        drag = 0.0 if vacuum else 0.5 * case.cd * case.area_m2 * MU/radius * np.interp(h, heights, densities)
        thrust = min(max_thrust, drag) if case.mode == MODES[1] else max_thrust
        return thrust, drag

    def rhs(t, y):
        h, fuel, phase = y
        radius = max(RE+1000*h, RE)
        mass = case.wet_kg - case.fuel_kg + max(fuel, 0)
        thrust, drag = forces(h)
        return [2*np.sqrt(radius**3/MU)*(thrust-drag)/mass/1000, -thrust/(G0*case.isp_s), np.sqrt(MU/radius**3)]

    def floor(t, y): return y[0]-120
    def target(t, y): return y[0]-case.target_km
    def empty(t, y): return y[1]
    events, reasons = [floor], ["120 km cutoff"]
    floor.direction = -1
    target.direction = 1
    empty.direction = -1
    if not passive and case.mode in (MODES[0], MODES[3]):
        events.append(target)
        reasons.append("Target reached")
    if max_thrust > 0:
        events.append(empty)
        reasons.append("Propellant exhausted")
    for event in events: event.terminal = True
    sol = solve_ivp(rhs,(0,case.days*DAY),[case.start_km,case.fuel_kg,0.0],
                    dense_output=True,events=events,rtol=2e-8,atol=[1e-7,1e-10,1e-6],
                    max_step=min(case.days*DAY/32,DAY))
    if not sol.success: raise ValueError(sol.message)
    status = next((label for label, hit in zip(reasons, sol.t_events) if len(hit)), "Time limit")
    t = np.linspace(0, sol.t[-1], 321)
    h, fuel, phase = sol.sol(t)
    fuel = np.clip(fuel,0,case.fuel_kg)
    thrust, drag = np.array([forces(v) for v in h]).T
    mass = case.wet_kg - case.fuel_kg + fuel
    data = pd.DataFrame(dict(time_days=t/DAY,altitude_km=h,phase_rad=phase,mass_kg=mass,
                             propellant_used_kg=case.fuel_kg-fuel,thrust_mN=thrust*1000,
                             drag_mN=drag*1000,delivered_dv_m_s=G0*case.isp_s*np.log(case.wet_kg/mass)))
    return data,status


def run_mission(case):
    track, end = propagate(case)
    if case.mode == MODES[2]:
        return track, end, None, None
    baseline, coast_end = propagate(replace(case, days=float(track.time_days.iloc[-1])), coast=True)
    return track, end, baseline, coast_end
