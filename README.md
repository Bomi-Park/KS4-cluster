# KS4 Red-Sequence Galaxy Cluster Pipeline

A modular, red-sequence–based galaxy cluster detection pipeline developed for  
the **KMTNet Synoptic Survey of the Southern Sky (KS4)**.

This repository provides a reproducible, configuration-driven framework to:
- select red-sequence galaxies
- construct density maps
- detect cluster candidates using SExtractor
- assign cluster members
- consolidate duplicated detections across redshift slices

The pipeline is designed for **large-area optical surveys** and is optimized
for batch processing over many fields and redshift bins.

---
<br>

## ✨ Features

- **7-step modular pipeline**
  1. Red-sequence galaxy selection  
  2. Density map construction  
  3. Density map visualization (optional)  
  4. Density map → FITS  
  5. SExtractor-based source detection  
  6. Cluster detection & member assignment  
  7. Diagnostic cluster images (optional)

- **Config-driven execution**
  - All parameters are controlled via YAML files
  - Easy experimentation and full reproducibility

- **Star–galaxy classification**
  - SExtractor CLASS_STAR
  - Optional Gaia DR3 cross-match

- **Batch execution utilities**
  - One field × many redshifts
  - One redshift × many fields
  - Multiprocessing support

- **Post-processing consolidation**
  - Merge duplicated cluster candidates across redshift slices
  - Optional member-overlap criterion

---
<br>

## 📁 Repository Structure
.  <br>
├── src/ <br>
│ └── ks4_rs/ <br>
│ ├── init.py <br>
│ ├── pipeline.py <br>
│ ├── runners.py <br>
│ ├── consolidation.py <br>
│ ├── classification.py <br>
│ ├── config.py <br>
│ ├── utils.py <br>
│ └── steps/ <br>
│ ├── step1_find_rs.py <br>
│ ├── step2_density_map.py <br>
│ ├── step3_plot_density.py <br>
│ ├── step4_make_fits.py <br>
│ ├── step5_source_extractor.py <br>
│ ├── step6_cluster_detect.py <br>
│ └── step7_cluster_images.py <br>
│ <br>
├── configs/ <br>
│ ├── pipeline_default.yaml <br>
│ ├── pipeline_test.yaml <br>
│ └── sex/ <br>
│ ├── default.sex <br>
│ ├── default.param <br>
│ ├── default.conv <br>
│ └── default.nnw <br>
│ <br>
├── scripts/ <br>
│ ├── run_pipeline.py <br>
│ └── run_single_field.py <br>
│ <br>
├── pyproject.toml <br>
├── requirements.txt <br>
├── .gitignore <br>
└── README.md <br>


---
<br>

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Bomi-Park/KS4-cluster.git
cd KS4-cluster
```

### 2. Create a Python environment (recommended)
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
**Option A (recommended, editable install):**
```bash
pip install -e .
```

**Option B (minimal, HPC-friendly):**
```bash
pip install -r requirements.txt
```

---
<br>

## 🚀 Quickstart (Test Run)
To quickly check that the pipeline runs end-to-end:

### 1. Edit loader functions

In the scripts below, fill in two small functions according to your environment:
- `loader(field_id)`
- `rs_model_lookup(z, info, config)`

Files:
- `scripts/run_pipeline.py`
- `scripts/run_single_field.py`

These functions are intentionally left user-defined to avoid hard-coding
survey-specific paths or BC03 model locations.

### 2. Run a single field at one redshift
```bash
python scripts/run_pipeline.py \
  --field KS4_TEST_001 \
  --z 0.35 \
  --config configs/pipeline_test.yaml
```

### 3. Run a single field over multiple redshifts
```bash
python scripts/run_single_field.py \
  --field KS4_TEST_001 \
  --zlist 0.2:0.6:0.05 \
  --config configs/pipeline_test.yaml
```

---
<br>

## ⚙️ Configuration

All pipeline parameters are defined in YAML files under configs/.

- `pipeline_default.yaml` <br>
Full-resolution, science-grade configuration
- `pipeline_test.yaml` <br>
Fast, low-resolution configuration for sanity checks

<br>Example (excerpt):
```yaml
density_map:
  nbins: 256
  smooth_sigma: 1.0

cluster_detect:
  member_radius_arcmin: 1.0
  min_members: 5
```
---
<br>

## 🧠 Design Philosophy

- **Code ≠ Parameters** <br>
Algorithms live in Python; parameters live in YAML.

- **Explicit over implicit** <br>
No hidden globals, no magic paths.

- **Survey-agnostic core** <br>
KS4-specific I/O is isolated in user-defined loaders.

- **Paper-ready structure** <br>
Pipeline steps map cleanly onto a Methods section.
---
<br>

## 📌 Notes on Data

This repository **does not include**:

- KS4 photometric catalogs
- Gaia catalogs
- BC03 stellar population synthesis outputs

Users must provide these data locally.

---
<br>

## 📝 Citation

If you use this pipeline in academic work, please cite:

> Park et al. (in prep.), <br>
KS4 Red Sequence Galaxy Cluster Survey

A formal software citation will be added upon publication.

---
<br>

## 📬 Contact

**Bomi Park** <br>
Ph.D. Candidate, Astronomy <br>
Seoul National University <br>

GitHub: https://github.com/Bomi-Park

---
<br>

## 📄 License

This project is released under the **MIT License**.
