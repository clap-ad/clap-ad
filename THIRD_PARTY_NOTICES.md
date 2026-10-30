# Third-Party Notices

This repository (built on the [LEAD](https://github.com/kesai-labs/lead)
codebase) vendors or depends on several third-party projects under
`3rd_party/`. This file lists their licenses. **This is provided for
convenience and is not legal advice** — please review the upstream licenses
yourself before any commercial use.

## MIT-licensed components

| Path | Project | License |
| :--- | :--- | :--- |
| `3rd_party/Bench2Drive/leaderboard`, `3rd_party/Bench2Drive/scenario_runner` | [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) vendored CARLA components | MIT |
| `3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard`, `.../scenario_runner` | Bench2Drive-Speed vendored CARLA components | MIT |
| `3rd_party/Bench2Drive-Speed/pdm_lite_speed/leaderboard`, `.../scenario_runner` | PDM-Lite vendored CARLA components | MIT |
| `3rd_party/leaderboard`, `3rd_party/leaderboard_autopilot` | [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard) | MIT |
| `3rd_party/scenario_runner`, `3rd_party/scenario_runner_autopilot` | [CARLA Scenario Runner](https://github.com/carla-simulator/scenario_runner) | MIT |
| `3rd_party/carla_route_generator` | [carla_route_generator](https://github.com/autonomousvision) | MIT |

These entries refer only to the listed MIT-licensed components, not their
parent benchmark trees. Preserve the upstream copyright and license notices.
The Fail2Drive root MIT license is included at `3rd_party/fail2drive/LICENSE`;
its nested CARLA components also retain their own license files.

## Apache-2.0-licensed components

| Path | Project | License |
| :--- | :--- | :--- |
| `3rd_party/navsim_workspace/navsimv1.1`, `navsimv2.2` | [NAVSIM](https://github.com/autonomousvision/navsim) (Univ. of Tübingen) | Apache License 2.0 |
| `3rd_party/Bench2Drive-Speed/pdm_lite_speed` | PDM-Lite | Apache License 2.0, except nested separately licensed CARLA subtrees |
| Released checkpoint image encoders | [timm/resnet34.a1_in1k](https://huggingface.co/timm/resnet34.a1_in1k) (CARLA) and [timm/regnety_032.ra_in1k](https://huggingface.co/timm/regnety_032.ra_in1k) (NAVSIM) pretrained weights | Apache License 2.0 |

The applicable Apache-2.0 `LICENSE` files are retained. Preserve any upstream
`NOTICE` file if one is included in a future source update.

Both released checkpoints include fine-tuned image encoders initialized from
these pretrained weights. Preserve their license and attribution when
redistributing the checkpoints.

## Benchmark materials under CC BY-NC-ND 4.0

The Bench2Drive project states that its assets and code are under CC BY-NC-ND
4.0 unless separately licensed (see
`3rd_party/Bench2Drive/LICENSE` and its README). Its nested
`leaderboard/` and `scenario_runner/` directories carry separate MIT licenses.
The 220 XML route files under `data/benchmark_routes/bench2drive/` are
benchmark-derived material; preserve their attribution and confirm the scope
of the applicable benchmark license before redistributing them.

`3rd_party/Bench2Drive-Speed` likewise contains the
[Bench2Drive-Speed](https://thinklab-sjtu.github.io/Bench2Drive-Speed/)
benchmark (Shao et al.) and its bundled TCP-Speed baseline
(`3rd_party/Bench2Drive-Speed/TCP`), which CLAP-AD does **not** use. The root
benchmark files, `Bench2Drive/` assets, and `TCP/` are marked CC BY-NC-ND 4.0,
with separately licensed nested components such as the `Bench2Drive/leaderboard`
and `pdm_lite_speed/` Apache-2.0 subtrees.

The multi-GPU evaluator at
`3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_multigpu.py`
is an adaptation of the CARLA MIT-licensed evaluator and resides in the
nested MIT-licensed `Bench2Drive/leaderboard` subtree. It retains the upstream
MIT notice. The separate `route_speed.py` helper is also under that subtree's
MIT license. This is distinct from the CC BY-NC-ND benchmark assets; confirm
that the included evaluator changes contain no copied CC BY-NC-ND material.

The benchmark materials remain subject to **Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0** (see
`3rd_party/Bench2Drive-Speed/LICENSE`). The license allows noncommercial
sharing of unadapted material with attribution; it does not grant permission
to use the material commercially or to share adapted material. Obtain
permission from the relevant rights holders for those uses. This notice is
not legal advice.

## NOT included: UniAD / VAD CLAP-AD variants (Bench2DriveZoo)

The paper also reports CLAP-AD results built on top of UniAD and VAD, which in
our internal setup live in a separate codebase,
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo), licensed
under **CC BY-NC-ND 4.0**. Because that license forbids redistributing
derivative works, and a CLAP-AD-modified UniAD/VAD planning head is exactly
that, **this repository does not include any UniAD/VAD code or checkpoints**.
The paper's UniAD/VAD numbers are still reported (as read-only reference
values) in the root README for completeness, with no code or weights attached — only the TFv6 (CARLA) and
LTFv6 (NAVSIM) implementations of CLAP-AD are released here. The applicable
license for LEAD-derived and CLAP-AD-modified code must be determined from the
root license together with the upstream notices and the rights holders'
authorization; this notice does not establish ownership.

## First-party utilities without a separate license

`3rd_party/typings/carla` (type stubs for the CARLA Python API) and
`3rd_party/vscode_pydata_viewer` (a small pickle viewer) are carried in the
LEAD codebase. Retain their upstream attribution and license notices; apply
this repository's MIT license only to original contributions whose rights
holders have authorized that license.

## CARLA simulator itself

`3rd_party/CARLA_0915` is not part of this repository (it is downloaded or
symlinked separately, see the root README) and is covered by its own license —
see [carla-simulator/carla](https://github.com/carla-simulator/carla).
