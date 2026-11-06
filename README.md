# kepler-exoplanet-hunter

**Vetting Kepler Transit Signals: Comparing Rule-Based Physics, Classical Machine Learning, and 1D CNNs to Find Planet Candidates**

---

### Quick Launch

<a href="https://colab.research.google.com/github/RileyMarek/kepler-exoplanet-hunter/blob/main/Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="38"/>
</a>

> **Direct Colab Link:**  
> `https://colab.research.google.com/github/RileyMarek/kepler-exoplanet-hunter/blob/main/Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb`

> **GitHub Repository:**  
> `https://github.com/RileyMarek/kepler-exoplanet-hunter`

---

## Team Workflow & Google Colab Sync Guide

This repository is synced directly with Google Colab so the whole team (Jackson, Taylor, Riley) can work without local environment headaches.

### How It Works Under the Hood
The "Open In Colab" link dynamically pulls the **latest commit on the `main` branch** of this GitHub repository every time it is opened. You will always be working with the most recent version of the notebook.

### Step-by-Step Team Editing Workflow

```
1. Click "Open In Colab"  ──>  2. Write / Test Code in Browser  ──>  3. File > "Save a copy in GitHub"
```

1. **Launch the Notebook:**
   - Click the **Open In Colab** badge above or use the direct link.
   - Colab will clone the live `main` branch into a fresh, GPU/TPU-capable cloud instance.

2. **Run & Edit Your Section:**
   - Run **Section 0** to install domain libraries (`lightkurve`, `astropy`) and mount our shared Google Drive folder (`/content/drive/MyDrive/KTDG_Kepler_Project/data/`).
   - Locate your designated section badge (`[CHUNK 1 - JACKSON]`, `[CHUNK 2 - TAYLOR]`, or `[CHUNK 3 - RILEY]`) and write your code.

3. **Sync Changes Back to GitHub (Crucial Step):**
   - Once your section runs cleanly, save your work directly back to GitHub from Colab:
     1. Click **File** in the top navigation bar.
     2. Select **Save a copy in GitHub...**
     3. Select Repository: `RileyMarek/kepler-exoplanet-hunter`
     4. Select Branch: `main`
     5. File path: `Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb`
     6. Enter a clear commit message (e.g., `feat: implement DR24 data cleaning and grouped split` or `feat: add lightkurve phase-folding and visualization 1`).
     7. Click **OK**.
   - Your updates are immediately committed to GitHub, and the next person who clicks the "Open in Colab" badge gets your updated code automatically.

4. **Team Collaboration Etiquette:**
   - Before starting a coding session, refresh the GitHub page or click the Colab link fresh to make sure you have the newest commit from your teammates.
   - If two people need to work at the exact same time, communicate which cells you are touching, or save with distinct commit notes to avoid overwriting each other's outputs.

---

## Midpoint Division of Responsibilities

| Team Member | Midpoint Role | Deliverables |
| :--- | :--- | :--- |
| **Jackson Lusk** | **Chunk 1:** Data Ingestion & Cleaning *(Target: Saturday AM)* | Filter unvetted `UNK` rows; map `PC` $\to 1$, `AFP`/`NTP` $\to 0$; isolate 12 physical features; median imputation; 70/15/15 split grouped strictly by `kepid` with 0% leakage assertion; export `train.csv`, `val.csv`, `test.csv` to Drive; plot class imbalance. |
| **Taylor Tran** | **Chunk 2:** Light Curves & Physics Baseline *(Target: Sat–Sun)* | Query MAST with `lightkurve`; remove cosmic-ray outliers; detrend stellar variability; phase-fold light curves; **Visualization 1** (U-shape Candidate vs. V-shape Eclipsing Binary); build & score deterministic Physics Baseline. |
| **Riley Marek** | **Chunk 3:** EDA & Classical ML *(Target: Sat–Sun)* | **Visualization 2** (SNR, depth, duration distributions by class); train class-weighted Logistic Regression; train class-weighted Random Forest; Gini feature importances plot; validation scoring & error analysis. |
| **All Members** | **Midpoint Synthesis** | Unified 3-model benchmark table on `val.csv`; false-alarm / missed-planet failure analysis; submit group discussion post. |

---

## Project Structure

```text
kepler-exoplanet-hunter/
├── .gitignore                                      # Ignores large datasets (*.csv, data/) & caches
├── README.md                                       # Workflow guide, quick launch & roles
└── Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb  # Primary collaborative Colab notebook
```
