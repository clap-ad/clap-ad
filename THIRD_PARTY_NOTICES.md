# Third-Party Notices

This repository builds on [LEAD](https://github.com/kesai-labs/lead) and
contains vendored components under `3rd_party/`. The root [LICENSE](LICENSE)
covers original CLAP-AD contributions, released checkpoints, and LEAD-derived
software only within the scope stated there. It does not replace the component
licenses below. This inventory is not legal advice.

## MIT-licensed components

| Path | Project | License |
| :--- | :--- | :--- |
| `3rd_party/Bench2Drive/leaderboard`, `3rd_party/Bench2Drive/scenario_runner` | [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) vendored CARLA components | MIT |
| `data/benchmark_routes/bench2drive` | Per-route copies of `3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml` | MIT, from the nested CARLA Leaderboard component |
| `3rd_party/Bench2Drive-Speed/Bench2Drive/leaderboard`, `.../scenario_runner` | Bench2Drive-Speed vendored CARLA components | MIT |
| `3rd_party/Bench2Drive-Speed/pdm_lite_speed/leaderboard`, `.../scenario_runner` | PDM-Lite vendored CARLA components | MIT |
| `3rd_party/leaderboard`, `3rd_party/leaderboard_autopilot` | [CARLA Leaderboard](https://github.com/carla-simulator/leaderboard) | MIT |
| `3rd_party/scenario_runner`, `3rd_party/scenario_runner_autopilot` | [CARLA Scenario Runner](https://github.com/carla-simulator/scenario_runner) | MIT |
| `3rd_party/fail2drive` | [Fail2Drive](https://github.com/autonomousvision/fail2drive) | MIT |
| `3rd_party/carla_route_generator` | [carla_route_generator](https://github.com/autonomousvision) | MIT |

These entries refer only to the listed MIT-licensed components, not their
parent benchmark trees. Preserve the upstream copyright and license notices.
The Bench2Drive route files are exact XML copies of the MIT-licensed aggregate
route file listed above. CLAP-AD's multi-GPU and route-merging changes are in
the nested MIT-licensed CARLA Leaderboard code. The modified merger utility is
kept in that MIT subtree; the CC BY-NC-ND-only upstream `tools/merge_route_json.py`
is unmodified. Fail2Drive's root MIT license is included, and its nested CARLA
components retain their own license files.

## Apache-2.0-licensed components

| Path | Project | License |
| :--- | :--- | :--- |
| `3rd_party/navsim_workspace/navsimv1.1`, `navsimv2.2` | [NAVSIM](https://github.com/autonomousvision/navsim) (Univ. of Tübingen) | Apache License 2.0 |
| `3rd_party/Bench2Drive-Speed/pdm_lite_speed` | PDM-Lite | Apache License 2.0, except nested separately licensed CARLA subtrees |
| Released checkpoint image encoders | [timm/resnet34.a1_in1k](https://huggingface.co/timm/resnet34.a1_in1k) (CARLA) and [timm/regnety_032.ra_in1k](https://huggingface.co/timm/regnety_032.ra_in1k) (NAVSIM) pretrained weights | Apache License 2.0 |

The Apache-2.0 `LICENSE` files for the vendored NAVSIM and PDM-Lite sources are
retained. Include the Apache-2.0 license and preserve any upstream `NOTICE`
file when redistributing pretrained timm weights or future source updates.

Both released checkpoints include fine-tuned image encoders initialized from
these pretrained weights. Preserve their license and attribution when
redistributing the checkpoints.

## Benchmark materials under CC BY-NC-ND 4.0

The Bench2Drive project states that its materials are under CC BY-NC-ND 4.0
unless separately licensed (see `3rd_party/Bench2Drive/LICENSE` and its
README). Its root-level documentation, assets, and tools remain under that
license; the nested `leaderboard/` and `scenario_runner/` directories have
separate MIT licenses. Bench2Drive materials are attributed to Xiaosong Jia,
Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, and Junchi Yan; see the upstream
[paper](https://arxiv.org/abs/2406.03877).

`3rd_party/Bench2Drive-Speed` likewise contains the
[Bench2Drive-Speed](https://thinklab-sjtu.github.io/Bench2Drive-Speed/)
benchmark (Shao et al.) and its bundled TCP-Speed baseline
(`3rd_party/Bench2Drive-Speed/TCP`), which CLAP-AD does **not** use. The root
documentation and assets, `Bench2Drive/` materials, and `TCP/` are marked
CC BY-NC-ND 4.0 unless a nested license says otherwise. The nested CARLA
`leaderboard/` and `scenario_runner/` trees are MIT; `pdm_lite_speed/` is
Apache-2.0 except for its nested separately licensed CARLA trees. Bench2Drive-
Speed is attributed to Yuqian Shao, Xiaosong Jia, Langechuan Liu, and Junchi
Yan; see the upstream [paper](https://arxiv.org/abs/2603.25672).

The multi-GPU evaluator and `route_speed.py` helper in the nested
`Bench2Drive/leaderboard` subtree are under that subtree's MIT license. The
multi-GPU evaluators and the changed standard evaluator retain their upstream
copyright and MIT headers. The CLAP-AD route-merging utility is kept in the
MIT-licensed `3rd_party/Bench2Drive/leaderboard` subtree. These code changes
do not change the license of the surrounding benchmark materials.

The CC BY-NC-ND materials are included unmodified and remain under their
upstream licenses; the copied license files and source acknowledgments are
retained. The Bench2Drive materials are from
[Thinklab-SJTU/Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive);
the Bench2Drive-Speed materials are from the
[official project](https://github.com/Thinklab-SJTU/Bench2Drive-Speed).
That license permits sharing unadapted material with attribution for
noncommercial purposes. It does not grant commercial-use rights or permission
to share adaptations. Obtain separate permission for those uses.

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

## LEAD-carried utilities

`3rd_party/typings/carla` (CARLA API type stubs) and
`3rd_party/vscode_pydata_viewer` (a pickle viewer) are carried from LEAD and
are covered by the upstream LEAD MIT notice retained in the root [LICENSE](LICENSE).
This does not replace any additional third-party rights that may apply to
generated API declarations or external assets.

## CARLA simulator itself

`3rd_party/CARLA_0915` is not part of this repository (it is downloaded or
symlinked separately, see the root README) and is covered by its own license —
see [carla-simulator/carla](https://github.com/carla-simulator/carla).
