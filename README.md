# kepler-exoplanet-hunter

**Vetting Kepler Transit Signals: Comparing Rule-Based Physics, Classical Machine Learning, and 1D CNNs to Find Planet Candidates**

---

### Quick Launch

<p align="center">
  <a href="https://colab.research.google.com/github/RileyMarek/kepler-exoplanet-hunter/blob/main/Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb" target="_blank" rel="noopener noreferrer">
    <img src="assets/launch-colab.svg" alt="Launch in Google Colab" width="320" height="54" />
  </a>
</p>

> **Direct Colab Link (New Tab):**  
> <a href="https://colab.research.google.com/github/RileyMarek/kepler-exoplanet-hunter/blob/main/Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb" target="_blank" rel="noopener noreferrer">https://colab.research.google.com/github/RileyMarek/kepler-exoplanet-hunter/blob/main/Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb</a>

> **GitHub Repository:**  
> `https://github.com/RileyMarek/kepler-exoplanet-hunter`

---

## Team Workflow & Google Colab Sync Guide

This repository is synced directly with Google Colab so the whole team (Riley, Jackson, Taylor) can run, edit, and sync without local setup friction.

### How It Works
The **Launch in Google Colab** button dynamically clones the **latest commit on the `main` branch** of this GitHub repository every time it is opened. You will always be working with the most recent version of the notebook.

### In-Browser Development Workflow

```
1. Click Launch Button  ──>  2. Run & Edit Assigned Section  ──>  3. File > "Save a copy in GitHub"
```

1. **Launch:** Click the green **Launch in Google Colab** button above.
2. **Setup:** Run **Section 0** to mount the shared Google Drive (`DATA_DIR`) and import dependencies.
3. **Edit Your Section:** Work within your designated section (`Section 1`, `Section 2`, or `Section 3`).
4. **Save Back to GitHub:**
   - Go to **File** $\to$ **Save a copy in GitHub...**
   - Select Repository: `RileyMarek/kepler-exoplanet-hunter`
   - Select Branch: `main`
   - File Path: `Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb`
   - Enter a descriptive commit message (e.g., `feat: implement DR24 data cleaning and grouped split`).
   - Click **OK**.
5. **Collaborator Access:** All team members added as repository collaborators can commit directly back to `main` from their browser.

---

## Midpoint Responsibilities

| Section | Main Contributor | Deliverables |
| :--- | :--- | :--- |
| **Section 1: Data Cleaning & Preprocessing** | **Jackson Lusk** *(Target: Saturday AM)* | Ingest DR24 TCE catalog; drop unvetted `UNK` rows; map targets (`PC` $\to 1$, `AFP`/`NTP` $\to 0$); select 10–12 physical features; median imputation; 70/15/15 split grouped strictly by `kepid` with 0% leakage assertion; export `train.csv`, `val.csv`, `test.csv`; plot class distribution. |
| **Section 2: Light Curves & Baseline** | **Taylor Tran** *(Target: Sat–Sun)* | Query MAST with `lightkurve`; outlier masking; stellar detrending; phase-folding; **Visualization 1** (U-shape Candidate vs. V-shape Eclipsing Binary); build & score deterministic Physics Baseline on `val.csv`. |
| **Section 3: EDA & Classical ML** | **Riley Marek** *(Target: Sat–Sun)* | **Visualization 2** (SNR, depth, duration distributions by class); train class-weighted Logistic Regression; train class-weighted Random Forest; Gini feature importances; score models on `val.csv`. |
| **Section 4: Midpoint Synthesis** | **All Contributors** | Unified 3-model benchmark table on `val.csv`; false-alarm & missed-planet error analysis. |
| **Section 5: Check-In Submission** | **All Contributors** | Formatted discussion post addressing all project check-in rubric criteria. |

---

## Project Structure

```text
kepler-exoplanet-hunter/
├── assets/
│   └── launch-colab.svg                            # Custom solid-green Colab launch button
├── .gitignore                                      # Ignores data artifacts (*.csv, data/) & checkpoints
├── README.md                                       # Workflow guide, quick launch & responsibilities
└── Main_Workspace_Vetting_Kepler_Transit_Signals.ipynb  # Primary collaborative Colab notebook
```
