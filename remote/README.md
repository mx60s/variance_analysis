# Remote collection and training: reviewed 2026-11-16

A bounded inference-only run (`modal_inference.py`) has been launched after a pre-launch estimate of under $1 compute and a $5 setup allowance. No training job has been launched. Modal authentication was verified with a read-only app listing; AWS STS authentication also succeeded. Google Cloud authorized-user configuration exists but was not validated. Credentials remain local and are not included in these files.

## Recommended first run

Use Modal for a bounded, single-L4 training/throughput pilot. `modal_pilot.py` is prepared and imports successfully, but its training image has not been built remotely and CUDA training with this pilot has not been tested. The separate inference runner has now executed successfully on an A10; this is not a training-throughput benchmark. The local `scripts/train_pilot.py` smoke test completed five optimizer steps and validation on the authors' public trajectories. This is a plumbing test, not a trained model or a reproduction of the paper.

The existing checkpoint is a 1,518,083-parameter network. Default model behavior remains exactly checkpoint-compatible. The pilot explicitly selects a new causal-normalization architecture: per-frame LayerNorm and GroupNorm replace operations that mix future frames into earlier predictions. This changes the architecture and requires new training. Preserve an unchanged upstream-model control for scientific comparisons.

The pilot uses ten-frame windows (nine next-frame targets), episode-disjoint train/validation folders, a fixed seed, finite-loss checking, gradient clipping, a time limit, saved optimizer state and run metadata. It intentionally uses the first window of each episode and a small fixed learning rate: it is not the final 200-epoch training recipe. Before full training, decide window sampling, dataset/world identity, normalization statistics, independent validation/test trajectories, learning-rate scheduling and model seeds. Resume and full-data training still need a production training runner.

## Cost estimate before launch

[Modal pricing](https://modal.com/pricing), checked 2026-11-16:

- L4: $0.000222/s = $0.7992/hour.
- Two physical CPU cores: $0.09432/hour.
- 8 GiB RAM: $0.063936/hour.
- Requested combined compute: approximately $0.9575/hour. CPU/RAM billing can exceed the reservation if usage is higher. Build time, storage and egress are separate; do not assume unused free credits.

Allow **$5 for the first pilot**. The function has a 30-minute timeout, no retries, one container maximum, and a 10-minute training-loop limit. This is a runtime bound, not a provider-enforced dollar cap. The first job should measure real throughput before authorizing a larger run.

Full training cannot yet be quoted: the exact training dataset and GPU throughput are not verified. Illustrative calculation only, at 100 ten-frame windows/second and 200 epochs:

| Windows per epoch | GPU hours per seed | Compute per seed at $0.9575/h |
|---:|---:|---:|
| 10,000 | 5.56 | $5.32 |
| 50,000 | 27.78 | $26.60 |
| 100,000 | 55.56 | $53.19 |

Multiply by seeds and add validation, collection, storage and failed-run overhead. A factor-of-two throughput difference doubles/halves these numbers. The Mac smoke test is not a reliable CUDA pricing benchmark.

After cost review, create a dedicated `variance-training-data` volume, upload verified PNG trajectory episodes to `/train` and `/validation`, and run:

```
modal run remote/modal_pilot.py --train /data/train --validation /data/validation
```

The wrapper uses `create_if_missing=False` so it cannot silently invent an empty data volume. A valid payment method and GPU availability still need confirmation at launch. Existing OpenRouter credentials are for model API access and are not a GPU training host.

## Minecraft collection

Prefer an isolated x86 Linux collection environment before attempting a world/version migration. Collection and training should be separate jobs with durable, checksummed data between them; Python 3.6 collection need not constrain modern PyTorch training.

The original Gornet environment notebook uses Malmo 0.37.0's Ubuntu 18.04/Python 3.6 binary, OpenJDK 8 and a virtual display. Its official release still exists. The Malmo repository is now archived. Reproducing that environment on Linux is plausible, but installation, asset downloads and frame rendering have NOT been tested in this task. The Mac has no Docker CLI installed. A remote CPU VM/container with Xvfb/Mesa is a reasonable compatibility pilot; actual rendering throughput must be measured. Do not assume a CUDA device automatically provides an OpenGL display.

[MineStudio](https://craftjarvis.github.io/MineStudio/overview/getting-started) is a maintained alternative with Python 3.10+, JDK 8 and Xvfb or VirtualGL support. It is not a drop-in replacement for this Malmo experiment. Test world import, textures, camera FOV, lighting, yaw conventions, observation timing and teleport commands before using it for the paper. A client that disables rendering cannot produce RGB training observations.

The user's [masters-thesis repository](https://github.com/mx60s/masters-thesis) contains `environments/small-world`, `Natural-Environment`, Malmo XML and random-walk/obstacle-avoidance/heading-control agents. The small-world XML uses the same 120..136 by 197..213 grid as this analysis and 64x64 images: strong evidence of relevance, not proof that the exact world/checkpoint pair has been recovered. The source path inside its XML is machine-specific and must be replaced at deployment.

Collection acceptance checks:

1. Record a world-save checksum, simulator/build versions, resource-pack and camera settings, trajectory seed and sampling clock.
2. Save commanded AND observed XYZ/yaw/pitch and synchronize them with the returned video frame; wait for fresh observations after teleport/turn commands. Existing fixed-pitch data has 44 frames across 41 locations with observed-yaw errors >0.01 degrees, maximum 5.73 degrees.
3. Require all 289 locations and 90 headings; record failure/retry counts. Never silently treat stale frames as requested headings.
4. Collect continuous held-out trajectories separately from training episodes, plus matched histories ending at evaluation states. Repeated static frames are a control, not a substitute for natural navigation history.
5. Archive raw RGB and metadata before extracting activations. Save feature IDs, model/checkpoint hash, preprocessing and software versions alongside every cache.

## Located sources and provenance

- Current Drive project: https://drive.google.com/drive/folders/18WTBV1cdqokFo-GOSGSQ-gUV4ua5QaFy
- Current Minecraft evaluation data: https://drive.google.com/drive/folders/1Ioq4nw3rB0VgzmtpfaPMudjqPrSLdPil
- Checkpoint: https://drive.google.com/file/d/1VwgqP87QXlY4o5z4EPlHs1UkdZJg8kET/view
- Archived project: https://drive.google.com/drive/folders/1HB766C_aBlchzXE6mY6JMSwhcHDsTrwO
- Candidate older train/validate parent: https://drive.google.com/drive/folders/16AOTvN_tfNIDmFBXJ06qQNqMyPWbmRDx (folder structure found; one sampled episode listing returned no files; exact training data not verified).
- Gornet source/data: https://github.com/jgornet/predictive-coding-recovers-maps at fed80b72925a4d3f3e9ec4b70f71ecaedec75c13.
- Luo source: https://github.com/don-tpanic/Space at 7a9028e09cd07e9b88c73dd8fcc43b2e3aab5b7f.
- Tanni source: https://github.com/Barry-lab/Publication_TanniDeCothiBarry2021 at 815398d6617af249e509ae193331f5d2927ddcf6.
- User's thesis source: https://github.com/mx60s/masters-thesis at e1956875f6fd1cff2dd02519b40c96973630e8de.
- Legacy simulator release: https://github.com/microsoft/malmo/releases/tag/0.37.0.

Public upstream PNG episodes were downloaded and can support a host/loader smoke test now. They should not be described as the exact data used for the current checkpoint. Its file contains weights, not sufficient training metadata to establish the original optimizer history or dataset identity.

## Revalidation inference

`prepare_remote_inference.py` builds a minimal stage with code, recorded feature IDs, reference activations and an explicit untrained ResNet state. `modal_inference.py` mounts only this stage, evaluation panoramas and the historical predictive checkpoint. It requests one L4 with A10 fallback, four CPU cores and 16 GiB RAM, with a 30-minute timeout and no retries (about $1.12/hour on L4 or $1.42/hour on A10 at the rates above, excluding setup/overhead). TF32 is disabled; a small cross-device activation check must pass for every model. Interpolated and native-grid EV results are returned in a ZIP; no raw credentials are included. The first setup app was stopped before a GPU container ran, to add an explicit untrained state rather than rely on cross-platform random initialization.

The full inference review was split into two bounded jobs so the first timeout could not interrupt the predictive-coder scores. Completed CNN outputs were backed up, verified locally, and the first app was stopped before the predictive-only job ran. L4-only requests remained queued with no containers; A10 fallback allocated successfully. This establishes an operational host for inference, not a validated Minecraft renderer or full training recipe.
