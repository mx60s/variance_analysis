---
name: hd-md-behavioural-coupling
description: In the Duszkiewicz sessions HD and movement direction are only weakly coupled behaviourally; MD sign convention verified correct
metadata: 
  node_type: memory
  type: project
  originSessionId: c9fe392c-1579-4978-80f7-7cbf9a20c0ef
  modified: 2026-09-13T21:46:39.988Z
---

Measured 2026-09-13 from raw Tracking_data.csv (100 Hz) for 4 Dusz sessions and from the 90x90 HD×MD occupancy of 21 sessions. P(|MD−HD|<30°) is ~0.25 at the notebook's 1 cm/s speed threshold and only 0.35–0.5 at ≥10 cm/s (independent baseline 0.17); mean resultant length of MD−HD ≈ 0.24. Every alternative sign/offset convention for MD (y flip, axis swap, HD mirrored) is worse, and position smoothing up to 1 s barely helps, so the current `get_md_from_xy` (arctan2(-dy, dx)) is right and the weak coupling is real behaviour, not a bug.

**Why:** Maggie wanted the "how much do HD and MD correlate" analysis tightened before claiming definitive HD-over-MD selectivity.

**How to apply:** Report the coupling with a speed threshold stated; weak coupling means the HD×MD joint map is well populated and EV_hd ≫ EV_md is not a sampling confound. See [[project-layout-two-copies]].
