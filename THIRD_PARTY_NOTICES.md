# Third-Party Notices

This repository (built on the [LEAD](https://github.com/kesai-labs/lead)
codebase) vendors or depends on several third-party projects under
`3rd_party/`. This file lists their licenses. **This is provided for
convenience and is not legal advice** — please review the upstream licenses
yourself before any commercial use.

## Vendored under `3rd_party/` (MIT-compatible)

| Path | Project | License |
| :--- | :--- | :--- |
| `3rd_party/Bench2Drive` | [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) leaderboard/scenario_runner | MIT |
| `3rd_party/leaderboard`, `3rd_party/leaderboard_autopilot` | [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard) | MIT |
| `3rd_party/scenario_runner`, `3rd_party/scenario_runner_autopilot` | [CARLA Scenario Runner](https://github.com/carla-simulator/scenario_runner) | MIT |
| `3rd_party/fail2drive` | [Fail2Drive](https://github.com/autonomousvision/fail2drive) leaderboard | MIT |
| `3rd_party/carla_route_generator` | [carla_route_generator](https://github.com/autonomousvision) | MIT |

These are compatible with this repository's MIT license and are redistributed
unmodified or with small compatibility patches, under their own `LICENSE`
files in each subdirectory.

## Vendored under `3rd_party/navsim_workspace/` (Apache-2.0)

| Path | Project | License |
| :--- | :--- | :--- |
| `navsimv1.1`, `navsimv2.2` | [NAVSIM](https://github.com/autonomousvision/navsim) (Univ. of Tübingen) | Apache License 2.0 |

Apache-2.0 is permissive and compatible with this repository's MIT license;
attribution and the Apache-2.0 `LICENSE`/`NOTICE` are preserved in each
subdirectory.

## Vendored under `3rd_party/Bench2Drive-Speed/` — ⚠️ CC BY-NC-ND 4.0

`3rd_party/Bench2Drive-Speed` vendors the
[Bench2Drive-Speed](https://thinklab-sjtu.github.io/Bench2Drive-Speed/) benchmark (Shao et al., desired-speed
evaluation) **unmodified**, under **Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0** (see
`3rd_party/Bench2Drive-Speed/LICENSE`). This includes its bundled TCP-Speed
baseline implementation (`3rd_party/Bench2Drive-Speed/TCP`), which CLAP-AD
does **not** use — the only part of this directory CLAP-AD depends on is the
`--bench2drive-speed` leaderboard evaluator harness
(`3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard`), used as-is to run the
48-route Bench2Drive-Speed benchmark.

**Implications:** CC BY-NC-ND forbids (a) commercial use and (b) distributing
modified/derivative versions of this specific directory. We do not modify it
and do not sublicense it — it remains under its own license, redistributed
verbatim for non-commercial research use only (consistent with how the
benchmark's own authors distribute it). If you need to use Bench2Drive-Speed
commercially, contact its authors directly.

## NOT included: UniAD / VAD CLAP-AD variants (Bench2DriveZoo)

The paper also reports CLAP-AD results built on top of UniAD and VAD, which in
our internal setup live in a separate codebase,
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo), licensed
under **CC BY-NC-ND 4.0**. Because that license forbids redistributing
derivative works, and a CLAP-AD-modified UniAD/VAD planning head is exactly
that, **this repository does not include any UniAD/VAD code or checkpoints**.
The paper's UniAD/VAD numbers are still reported (as read-only reference
values) in the root README for completeness, with no code or weights attached — only the TFv6 (CARLA) and
LTFv6 (NAVSIM) implementations of CLAP-AD are released here, both original
LEAD/CLAP-AD code under this repository's own MIT license.

## First-party utilities without a separate license

`3rd_party/typings/carla` (type stubs for the CARLA Python API) and
`3rd_party/vscode_pydata_viewer` (a small pickle viewer) are authored by the
LEAD/CLAP-AD project and are covered by this repository's own
[LICENSE](LICENSE) (MIT).

## CARLA simulator itself

`3rd_party/CARLA_0915` is not part of this repository (it is downloaded or
symlinked separately, see the root README) and is covered by its own license —
see [carla-simulator/carla](https://github.com/carla-simulator/carla).
