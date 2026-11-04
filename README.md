# EP Mission Design Studio

Extract the complete project. In the folder containing `app.py`, run the following with Python 3.12.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## First transfer

Open **Transfer design**, choose **LEO to GEO**, then select **SPT-140** and set installed units to two. **Supply full cluster power** assigns the two nominal inputs. The default orbit departs at 400 km and 28.5 degrees and arrives at GEO altitude and zero inclination. Submit the form to calculate the transfer. To enforce a smaller power budget, clear the full-power box and enter the available thruster input. Changes take effect in the result when you press **Calculate transfer**.

The result includes required and delivered delta-v, elapsed and firing time, propellant use, initial and final steering yaw, margins and the reason the calculation stopped. Download the GIF or the complete run, including the numerical trajectory and inputs. The six-system comparison holds the bank geometry, installed and spare counts, duty and available power fixed while substituting one model throughout the cluster.

## Workspaces

| Workspace | Calculation or content |
| --- | --- |
| Transfer design | Edelbaum circular-orbit sizing with plane change, or planar two-body Cartesian propagation. Presets cover LEO–GEO, navigation orbit, lunar distance, lowering, inclination changes, GTO–GEO and heliocentric Earth–Mars and Earth–Venus radius changes. Custom Earth and solar endpoints are supported. |
| Mission design | Drag-limited VLEO reboost, station keeping and passive decay with the archived NRLMSISE-00 tables. Includes the earlier ideal circular transfer and unpowered baseline. |
| Thruster explorer | Original 390-model catalogue with evidence, source links and performance comparisons. |
| Reference animations | 100 distinct assets. The 14 clips in the supplied EP package have three project-specific source downloads containing original scripts, inputs and final frames. Seven matched existing videos by SHA-256. Uploaded collection names expose every clip, including matches. |

## Cluster model

Choose up to six independently configured banks. Each can use any eligible record in the twelve catalogue families. The six quick picks are SPT-140, T6, RIT-2X, MR-510, RIT-22 and RIT-10-EVO. Calculations use the identified **database nominal point**. The supplied films use the operating points described in their original README. Their trajectories can therefore differ from a new calculation.

Power is allocated in bank order. Cold spares consume zero firing power. Parallel units can fire individually. Canted units fire as complete, azimuthally symmetric groups of two, three, four or six. Axial thrust is the sum of unit thrust times the cosine of cant. Mass flow uses the full unit thrust divided by its effective exhaust velocity. The cluster effective Isp follows from total axial thrust and total mass flow. The entered wet mass includes the installed hardware, tanks and propellant. Pulsed devices use equivalent mean operating points. Mixed-bank tank or PPU compatibility, plume impingement, attitude torques and packaging require separate analysis.

## Methods and limits

**Edelbaum** uses the chord between circular speeds in the transformed plane with angular separation πΔi/2. Inclinations share a node. The implemented domain is |Δi| < 2 radians. The mass-varying clock uses constant nominal thrust, effective Isp and duty; the predicted circular radius and inclination are reconstructed along the chord. Sizing assumes fixed input power. Budget-limited cases stop at the available time or fuel. A screen flags a mean fractional speed change above 5% per revolution. The coplanar Hohmann comparison includes only the radius change. [Edelbaum equation and domain](https://stax.strath.ac.uk/downloads/f1881m11v)

**Planar RK4** uses the guidance functions from the supplied lowthrust.py and integrates Cartesian position, velocity and mass. It retains the guidance effectivity threshold of 0.3 for Earth and 0.7 for the Sun. The circular target tolerances are 1% in semi-major axis and eccentricity below 0.01. The duration, propellant, dynamics and integration-step limits have separate status values. The optional heliocentric solar case multiplies mean thrust and flow by min(1, (r_departure/r)²), assuming duty modulation and energy buffering within the selected nominal capacity. A plane change requires the circular Edelbaum method. GTO starts in an eccentric ellipse. Earth transfer cases outside the VLEO workspace use vacuum dynamics.

**Mission endpoints.** Lunar distance means Earth-centred orbital radius. Heliocentric endpoints are circular orbit radii around the Sun. Arrival time and propellant are transfer-screening quantities; actual encounters require ephemerides, launch geometry and capture mechanics. Coordinates in the CSV use a common inertial node at zero longitude. The replay samples mission time and can skip revolutions. The Earth reference radius is 6378.137 km for transfer calculations and 6371 km in the earlier VLEO model, following each source project.

**Testing.** To run the regression suite, install pytest and run `python -m pytest -q` in this folder. The first Cartesian propagation compiles the dynamics with Numba.

## Original database documentation


# EP Technology Explorer

A deployment-ready Streamlit interface for the 390-model electric-propulsion database.

## What the app provides

- A searchable browser with filters for thruster type, acceleration mechanism, propellant, country, power, readiness and evidence status.
- A thrust–specific impulse trade-space plot with equal marker sizes.
- Side-by-side comparison of up to five thruster models.
- A model profile with performance parameters, readiness basis, source provenance and direct links where available.
- Evidence-status and physics-reconciliation summaries.
- Downloads for the filtered table and the full research workbook.

The default view contains the 246 source-verified records. All 390 records remain available by changing the Evidence status filter. “Source verified” means that the cited document was inspected; it does not mean the thruster was independently laboratory-tested by the database authors.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Deploy with Streamlit Community Cloud

1. Upload this entire folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the entry point to `app.py`.
4. Deploy. No secrets or environment variables are required.

Keep the `data` directory beside `app.py`; the app loads both packaged data files with relative paths.

## Database scope and validation

The release contains 390 distinct thruster-model records spanning 12 thruster types and 22 countries. Every record includes a citation. Evidence coverage is:

| Evidence status | Records | Meaning |
|---|---:|---|
| Source verified | 246 | Cited document inspected and stored value checked or assembled from it |
| Manufacturer data | 49 | Developer or product literature, without primary-source confirmation in this audit |
| Compiled source | 75 | Secondary catalogue or review; primary document not used to confirm the value |
| Source not inspected | 20 | Citation stored but document not inspected |

The app intentionally keeps evidence status separate from technology readiness. TRL indicates documented maturity; it is not a blanket validation certificate for every stored parameter.

## Project structure

```text
EP_Thruster_Explorer/
├── .streamlit/config.toml
├── app.py
├── data/
│   ├── ep_thruster_models_v1.csv
│   └── EP_Thruster_Database_390_Models_v1.0.0.xlsx
├── requirements.txt
├── tests/
│   ├── test_app.py
│   └── test_data_integrity.py
└── VALIDATION.md
```
