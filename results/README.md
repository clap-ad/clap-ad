# CLAP-AD result files

Structured summaries of the numbers reported in the CLAP-AD paper, provided so
they can be diffed/plotted programmatically instead of re-typed from the PDF.
These are **not raw simulator logs** — they are transcriptions of the paper's
tables. To reproduce comparable numbers yourself, see the root
[README](../README.md#quickstart) (CARLA / Bench2Drive, Bench2Drive-Speed) and
[3rd_party/navsim_workspace/README.md](../3rd_party/navsim_workspace/README.md)
(NAVSIM v2 navhard).

| File | Paper table | Benchmark |
| :--- | :--- | :--- |
| [carla/main_table_bench2drive220.json](carla/main_table_bench2drive220.json) | Table 1 | Bench2Drive, 220 routes |
| [carla/b2d_speed_48routes.json](carla/b2d_speed_48routes.json) | Table 2 | Bench2Drive-Speed, 48 routes |
| [carla/uniad_ablation_dev10.json](carla/uniad_ablation_dev10.json) | Ablation table | Dev10 (10-route subset) |
| [navsim/navsim_v2_navhard.csv](navsim/navsim_v2_navhard.csv) | Table 3 | NAVSIM v2, navhard |
| [carla/route_results/tfv6_clapad_bench2drive220_merged.json](carla/route_results/tfv6_clapad_bench2drive220_merged.json) | Table 1 (TFv6 + CLAP-AD row) | Per-route `checkpoint_endpoint`-style leaderboard output for the actual run behind the 92.30 DS / 75.00 SR number (aggregate recomputes to 92.28/75.00) |

Rows/entries for methods built on the separately-licensed
[Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo) codebase
(UniAD, VAD, and their CLAP-AD variants) are annotated as such — this repository
only ships code and checkpoints for the TFv6 (CARLA) and LTFv6 (NAVSIM)
backbones. See the root [LICENSE](../LICENSE) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for details.

## Per-route raw result files

[carla/route_results/](carla/route_results/) additionally holds raw, per-route
leaderboard output (one record per route: driving score, infractions,
scenario/town, weather) for runs we could positively identify in this
environment as matching a paper-reported aggregate to within rounding. Only
one such match was found so far (TFv6 + CLAP-AD on Bench2Drive-220); see the
task notes for other candidate runs that did not match closely enough to
include with confidence.
