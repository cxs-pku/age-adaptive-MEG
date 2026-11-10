dHCP neonatal atlas geometry sensitivity analysis
===============================================

Purpose
-------
This calculation checks whether replacing the idealized spherical cortical
source shape with an openly released neonatal cortical atlas changes the
source-pattern metrics used in the manuscript. It is an atlas geometry
sensitivity analysis. It is not a subject-specific MRI/BEM forward model, an
empirical MEG analysis, or a pediatric localization-accuracy estimate.

Public input
------------
The input archive is the dHCP symmetric neonatal surface template:
https://biomedic.doc.ic.ac.uk/brain-development/downloads/dhcpSym_template.zip
The selected files are week-40 symmetric 32k pial surfaces for the left and
right hemispheres. They contain 32,492 vertices and 64,980 triangles per
hemisphere. The atlas is described by Edwards et al. (2022), DOI
10.3389/fnins.2022.886772, and its neonatal surface processing by Makropoulos
et al. (2018), DOI 10.1016/j.neuroimage.2018.01.054.

Run
---
From the v2 project root:

  python "2-revision v2/codes and outputs/codes/dhcp_atlas_geometry.py"

The script uses Python 3.9, NumPy, and nibabel. It writes
`outputs/dhcp_atlas_geometry_summary.csv` and
`outputs/dhcp_atlas_geometry_parameters.json`.

Model choices
-------------
The 400 pial vertices are sampled with seed 20261026 from the superior cap
covered by the Lambert template. Vertex normals are computed from pial
triangles, with local hemisphere indices offset before concatenation. Source
positions are scaled so the sampled atlas median radius matches the modeled
neonatal cortical radius (42.2351 mm). Point sensors use the Biot-Savart
kernel of the main model, K=25, 41, or 61, and shell radii 58.4 mm (neonatal
age-adaptive) or 91.5 mm (adult-helmet radius applied to the neonatal
anatomy). Channel rows are divided by their RMS across the selected source
patterns before computing effective rank and nearest-source pattern distance.

Interpretation
--------------
The output quantifies how atlas-shaped cortical geometry interacts with the
two shell radii. It does not establish age-general pediatric performance. No
scalp/skull surfaces, conductor model, raw MEG, empirical noise, or inverse
reconstruction is used. The GA40 atlas is a population template rather than a
subject-specific MRI.
