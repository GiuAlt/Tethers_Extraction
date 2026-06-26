# Tethers_Extraction

**Automated extraction and quantification of membrane tether forces from AFM retract curves**

When an AFM tip retracts from a cell surface, membrane tethers form — thin lipid tubes pulled out as the tip separates. Their rupture force and length encode membrane tension and cytoskeleton–membrane adhesion energy. This pipeline processes raw JPK force files, identifies tether events in the retract segment, and extracts quantitative force parameters per cell condition and timepoint.

---

## Pipeline overview

```
.jpk-force files
        │
        ▼
 T_UploadData.py          ← Load raw AFM data, extract approach/retract segments,
 (data ingestion)            baseline-correct retract curves, compile into DataFrame
        │
        ▼
 T_Baseline.py            ← Additional baseline processing and signal quality filtering
        │
        ▼
 T_Quantification1.py     ← Detect tether events, extract force and length per event
        │
        ▼
 T_ResultsPlotting.py     ← Visualise tether force distributions per condition
```

---

## Scripts

| Script | Description |
|---|---|
| `T_UploadData.py` | Reads `.jpk-force` files using `jpkfile`. Extracts height and vDeflection signals from approach and retract segments. Converts raw deflection to force (nN) using cantilever spring constant `k`. Applies linear baseline correction to retract curves. Outputs a compiled DataFrame with per-curve metadata (cell type, drug treatment, timepoint, cell number). |
| `T_Baseline.py` | Secondary baseline correction and pre-processing pass. Ensures signal quality before quantification. |
| `T_Quantification1.py` | Detects tether rupture events in baseline-corrected retract curves. Extracts peak force (nN) and tether length (µm) per event. Aggregates statistics per cell and condition. |
| `T_ResultsPlotting.py` | Produces force distribution plots (histograms, box plots) comparing tether forces across experimental conditions and timepoints. |

---

## Methods

**Input format:** JPK `.jpk-force` files from a JPK NanoWizard AFM. File naming encodes condition, timepoint, and cell number (e.g. `T4.5C1-...`), parsed automatically.

**Cantilever calibration:** Spring constant `k` set manually per experiment (typical values: 0.004–0.006 N/m for soft cantilevers used on live cells).

**Segment extraction:** Approach (segment 0) and retract (segment 2) are extracted separately. Tether analysis is performed on the retract segment only.

**Baseline correction:** Linear fit to the far-from-contact region of the retract curve; subtracted from the full signal to remove cantilever drift.

**Tether detection:** Tether rupture events appear as discrete force steps (negative force excursions) in the retract curve. Events are identified by threshold crossing and peak extraction.

---

## Setup

```bash
git clone https://github.com/GiuAlt/Tethers_Extraction.git
cd Tethers_Extraction
pip install jpkfile numpy pandas scipy matplotlib
```

> **Note:** Raw `.jpk-force` data files are not included. Set the working directory in `T_UploadData.py` to your data folder and adjust the spring constant `k` to match your cantilever calibration.

---

## Context

Developed during my PhD in Biophysics at ETH Zurich as part of a study on membrane mechanics in cancer cells under pharmacological perturbation. Tether force measurements were used alongside Hertz model fits (see [Hertz_Fits](https://github.com/GiuAlt/Hertz_Fits)) to characterise cell mechanical phenotype.

---

*Giulia Ammirati · [github.com/GiuAlt](https://github.com/GiuAlt) · ETH Zurich, 2024*
