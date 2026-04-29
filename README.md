# FIESTA Galaxy workflows

Galaxy workflows that wire the four FIESTA Galaxy tools
([scattering-stacking](https://github.com/NordicESMhub/galaxy-tools/pull/79),
[planktonclas-inference](https://github.com/NordicESMhub/galaxy-tools/pull/80),
[foscat-features](https://github.com/NordicESMhub/galaxy-tools/pull/81),
[foscat-synthesis](https://github.com/NordicESMhub/galaxy-tools/pull/82))
together to reproduce each [FIESTA-OSCARS](https://fairease.eu/) chain
end-to-end.

## Structure

```
workflows/
├── decrop-reproduction/      # CNN baseline reproduction (planktonclas-inference × 1)
├── astro-synthesis/          # Cosmological LSS map synthesis (foscat-synthesis pure mode)
├── sst-gap-filling/          # SST cloud gap-filling (foscat-synthesis HEALPix mode)
├── sst-wgs84-gap-filling/    # SST gap-filling on WGS84 ellipsoid (foscat-synthesis WGS84 mode)
└── bio-stacking/             # Plankton classification with scattering stacking (3-tool DAG)
```

Each workflow directory contains:
- `*.gxwf.yml` — the gxformat2 workflow definition
- `*.gxwf-tests.yml` — planemo test cases (Tier 1 smoke + optional Tier 2 canonical)
- `README.md` — what chain this reproduces, expected results, how to run
- `test-data/` — small synthetic fixtures for the smoke test

## Two-tier verification

- **Tier 1 (smoke):** runs the workflow on tiny synthetic fixtures. Verifies
  the DAG executes correctly. Numbers are gibberish (untrained models / random
  fields) but the wiring is exercised end-to-end. Fast (~10 min per workflow
  on Apple Silicon).
- **Tier 2 (canonical):** runs the workflow on real FIESTA datasets and
  verifies outputs match the published nanopub numbers. Slow (~hour per
  workflow). Driver script in `scripts/run-canonical.sh`.

## How to run locally

Smoke test:
```bash
planemo test \
    --extra_tools /Users/annef/Documents/ScienceLive/galaxy-tools/tools \
    workflows/<chain>/<chain>.gxwf.yml
```

Canonical reproduction (requires real datasets — see per-workflow README):
```bash
scripts/run-canonical.sh <chain>
```

## FIESTA chain reference

| Chain | Tools used | Canonical numbers (from FORRT nanopubs) |
|---|---|---|
| decrop-reproduction | planktonclas-inference | CNN top-1 = 0.8634 on Decrop's test split |
| bio-stacking | foscat-features × 3, planktonclas-inference × 2, scattering-stacking | Stacked rare-recall = 0.5608 (+8.4 percentage points vs CNN) |
| astro-synthesis | foscat-synthesis (pure synthesis on HEALPix) | LSS map synthesis matches power-spectrum and scattering stats |
| sst-gap-filling | foscat-synthesis (HEALPix gap-filling) | RMSE improvement vs spherical-harmonics baseline |
| sst-wgs84-gap-filling | foscat-synthesis (WGS84-resampled HEALPix gap-filling) | RMSE comparison sphere vs WGS84 |

## Downstream: Galaxy Training Network

Once Tier 2 verified, these workflows will be ported into
[Galaxy Training Network](https://training.galaxyproject.org/) tutorials.
This repo is the dev/test staging ground; GTN is the publication channel.

## License

Released under the [MIT License](LICENSE) — same as the FIESTA chain
repos this workflow set reproduces.
