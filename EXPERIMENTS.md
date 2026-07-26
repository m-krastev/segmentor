# Navigator experiments

This is the append-only experiment ledger for Navigator work. Every training,
evaluation, diagnostic, and ablation must record the code revision, data split,
command or service, acceptance metrics, and outcome here.

## Acceptance criteria

- Mean held-out path Dice (`validation/avg_dice`) of at least `0.40`.
- Complete start-to-end traversal, reported independently as endpoint reach
  rate and jointly as traversal success. Reward alone is never an acceptance
  metric.
- Deterministic held-out evaluation uses the Beta policy mean unless an
  experiment explicitly states otherwise.
- Training and validation case IDs use the seed-42 deterministic 90/10 split.
- Failed and negative experiments stay in this file.

## 2026-07-26 — N0: nnU-Net scratch PPO baseline

- Revision: `e71b9eb349564b4685086b9ee76d869182977fe2`
- Service: `navigator-nnunet-scratch-million.service`
- TensorBoard run: `20260726-143037-855781`
- Data: 585 matched cases from `nnUNet_raw`; deterministic seed-42 90/10 split.
- Configuration: scratch PPO, 1,000,000 target steps, 1,024 frames/batch,
  batch size 128, four PPO epochs, learning rate `5e-5`, 1.5 mm voxels,
  24 mm patch, 6 mm maximum action, 6 mm path radius, 1,024-step horizon.
- Health checks:
  - 24 unit/smoke tests passed.
  - CUDA and CuCIM were active.
  - First checkpoint, `checkpoint_25600.pth`, was written successfully.
  - At 46,080 steps all optimization losses were finite:
    policy `-0.00306`, value `0.35825`, KL `0.00438`.
- Numeric outcome when stopped at 62,464 steps:
  - terminal Dice (`train/final_coverage`): `0.24460`
  - traversal success: `0.0`
  - episode step count: `1024`
- Preserved checkpoints: `checkpoint_25600.pth` and
  `checkpoint_51200.pth`.
- Verdict: **failed baseline / stop early**. Dice regressed and no traversal
  succeeded. The final single training episode's higher Dice is too noisy to
  establish progress, and no held-out validation had occurred. Do not select
  by reward or by one training episode.

## 2026-07-26 — D0: geodesic reachability diagnostic

- Revision: `e71b9eb349564b4685086b9ee76d869182977fe2`
- Motivation: baseline `max_gdt_achieved` values were 14–24 million, far above
  plausible bowel path lengths.
- Procedure:
  - Read cached start/end points and GDT values for the first 29 complete
    cached cases.
  - Check start/end connectivity with 26-neighbor connected components for 30
    cached cases.
  - Compare the existing fast-marching field with a mask-constrained
    `skimage.graph.MCP_Geometric` field on `s0001`, `s0038`, and `s0486`.
- Results:
  - Existing GDT start-to-end values exceeded `1e5` in 29/29 cases; median
    `16,009,920`.
  - Start and end were in the same 26-neighbor component in 24/30 cases.
  - Mask-constrained distances:
    - `s0001`: `240.62 mm`, reachable, 0.31 s.
    - `s0038`: `175.82 mm`, reachable, 0.49 s.
    - `s0486`: unreachable, correctly reported as infinite, 0.13 s.
- Finding: the current fast-marching implementation gives background a speed
  of `1e-5`; it therefore leaks through forbidden background and supplies a
  false progress potential. Some cases are also genuinely disconnected.
- Decision: replace the GDT with a mask-constrained 26-neighbor transform,
  explicitly handle unreachable anatomical endpoints, regenerate versioned
  caches, and establish an oracle-path Dice/traversal ceiling before further
  PPO tuning.

## 2026-07-26 — T0: local post-GDT test attempt

- Command: `uv run --project . --no-sync python -m unittest discover -s tests -v`
- Result: **environment failure before test collection**. The local worktree
  environment resolved to a bare Homebrew Python 3.14 and lacked NumPy, Torch,
  pytest, and the installed `navigator` package.
- Decision: do not change the local environment. Run the authoritative suite
  with `uv run --no-sync` in the isolated commander worktree, whose CUDA
  environment previously passed all 24 tests.

## 2026-07-26 — T1: post-GDT commander test suite

- Environment: isolated commander worktree, existing CUDA environment.
- Command: `uv run --no-sync pytest -q tests`
- Result: **28 passed** in 4.56 s. There were 18 upstream deprecation
  warnings and no failures.

## 2026-07-26 — A0: mask-constrained shortest-path oracle

- Cases: `s0001`, `s0038`, `s0190`, `s0546`, `s0918`.
- Output: `results/navigator_nnunet/oracle-mcp26-v1/summary.json`
- Method: greedy realizable descent on the corrected MCP26 distance field;
  no learned policy and no stochastic action selection.
- Results:
  - endpoint reach rate: `1.00`
  - mean Dice: `0.02462`
  - minimum Dice: `0.02005`
  - joint traversal success at Dice >= 0.40: `0.00`
  - paths reached the endpoint in 18–31 actions.
- Verdict: **failed geometry control**. The shortest route cuts between
  spatially touching bowel loops. Reaching the anatomical endpoint is not
  equivalent to traversing the bowel volume.

## 2026-07-26 — A0.1: medial-skeleton tube Dice ceiling

- Cases: same five cases as A0.
- Method: Lee 3-D medial skeleton of the complete small-bowel mask, dilated
  with the same L1-ball geometry as the environment. This is a geometric
  ceiling, not a learned result and not yet a single ordered traversal.
- Mean Dice by tube radius:
  - 3 vox / 4.5 mm: `0.2434`
  - 4 vox / 6.0 mm (current): `0.3537`, minimum `0.2962`
  - 5 vox / 7.5 mm: `0.4605`, minimum `0.3900`
  - 6 vox / 9.0 mm: `0.5537`, minimum `0.4773`
- Finding: the requested 0.40 Dice is not attainable by the complete medial
  skeleton on 3/5 cases under the current 6 mm evaluation tube. A fixed 9 mm
  tube clears 0.40 on all five, but changing evaluation geometry must not be
  presented as learning progress.
- Decision: construct and test a continuous skeleton-covering start-to-end
  oracle. Keep Dice exact and report the tube radius explicitly; do not tune
  or select by reward.

## 2026-07-26 — T2: initial skeleton-route focused test

- Command: focused pytest run for the skeleton oracle, GDT, and environment.
- Result: **failed / terminated**. The synthetic route test consumed CPU and
  memory without completing; the process reached roughly 79% of commander RAM
  and was killed. Memory returned to normal.
- Initial suspected cause: calling `MCP_Geometric.traceback` when connector
  start and end were the same skeleton voxel.
- Initial fix: return the singleton connector directly for identical
  endpoints before invoking MCP. The safety-bounded rerun below showed that
  this was necessary but not the cause of the unbounded walk.

## 2026-07-26 — T3: skeleton-route test after singleton-connector guard

- Command: skeleton-oracle pytest under `timeout 60`.
- Result: **failed / timed out** after 60 s. The process again expanded the
  route until the safety timeout killed it; peak observed RAM was about 40%.
- Cause: a lazy generator for child nodes closed over the loop's mutable
  `node` variable. The excluded parent changed as the stack advanced, allowing
  the undirected tree walk to revisit its parent indefinitely.
- Fix: materialize each child tuple before pushing the DFS stack frame.

## 2026-07-26 — T4: skeleton-route test after DFS fix

- Result: the disconnected control passed and the route completed immediately;
  one assertion failed (`0.96` Dice versus an incorrectly expected exact
  `1.00`).
- Cause: Lee skeletonization legitimately removes a redundant junction voxel
  from the one-voxel synthetic T mask.
- Fix: retain the exact skeleton-coverage assertion and require greater than
  `0.90` undilated Dice instead of equality with the unskeletonized mask.

## 2026-07-26 — T5: focused skeleton-oracle suite

- Command: focused pytest for skeleton oracle, GDT, and environment under a
  120-second safety timeout.
- Result: **13 passed** in 2.80 s; 14 upstream TorchScript deprecation warnings.

## 2026-07-26 — A0.2: executable skeleton oracle, 6 mm tube

- Cases: same five real cases as A0.
- Output: `results/navigator_nnunet/oracle-skeleton-r6-v1/summary.json`
- Result:
  - endpoint reach rate: `1.00`
  - mean Dice: `0.36203`
  - joint traversal success at Dice >= 0.40: `0.40`
  - executed path length: 638–848 actions, within the 1,024-step horizon.
- Verdict: topology-preserving traversal works end to end, but the original
  6 mm tube cannot meet the requested Dice consistently.

## 2026-07-26 — A0.3: initial executable skeleton oracle, 9 mm tube

- Initial result: mean Dice `0.49480`, endpoint reach `1.00`, joint traversal
  success `0.80`.
- Invalid detail: `s0038` stopped after 302 actions with Dice `0.32191`
  because the evaluator treated passing within the larger endpoint radius as a
  reason to stop even though the route and success condition were incomplete.
- Fix: a skeleton run now stops only on environment success or after both the
  dense route and endpoint are complete. The geodesic control still stops at
  its first endpoint reach by design.

## 2026-07-26 — A0.4: corrected executable skeleton oracle, 9 mm tube

- Cases: same five real cases as A0.
- Output: `results/navigator_nnunet/oracle-skeleton-r9-v2/summary.json`
- Result:
  - endpoint reach rate: `1.00`
  - mean Dice: `0.55745`
  - minimum Dice: `0.46730`
  - joint traversal success at Dice >= 0.40: `1.00`
  - executed path length: 638–847 actions.
- Verdict: **geometry control passes**. This establishes that a continuous,
  executable route can meet the requested numeric and end-to-end criteria
  within the existing horizon. It is an oracle result, not learned progress.
- Caveat: the fixed evaluation tube is 9 mm rather than the original 6 mm.
  Both radii remain reported, and this change must not be conflated with model
  improvement.

## 2026-07-26 — A1 setup: held-out skeleton-imitation ablation

- Cohort: `scripts/reachable_actual_cached_v1.cases` contains 20 cases
  from the first 30 cached cases whose anatomical endpoints are 26-neighbor
  connected and whose small-bowel mask contains at least 100,000 voxels.
- This is a preliminary ablation cohort, not the final benchmark. It excludes
  genuinely impossible/disconnected and obviously tiny masks before splitting,
  without using learned-policy outcomes.
- Plan: deterministic seed-42 split; segmentation-derived skeleton routes are
  generated only as training demonstrations. Held-out validation remains a
  deterministic policy rollout with exact Dice, endpoint reach, and joint
  traversal success.

## 2026-07-26 — A1: one-epoch skeleton imitation plus one PPO batch

- Service: `navigator-ablation-a1-bc1-r9.service`
- Cases: 18 train / 2 held out using seed 42.
- Held-out cases: `s0918`, `s0190`.
- Configuration: one pure-expert behavior-cloning epoch, 9 mm path tube, then
  1,024 PPO frames (one rollout batch / four optimizer epochs).
- Demonstration health: generated experts completed 18/18 training routes;
  11,632 labeled steps; BC loss `1.9426`.
- Policy-only held-out results:
  - immediately after BC: mean Dice `0.15`, traversal success `0.00`
  - after one PPO batch: mean Dice `0.0729`, traversal success `0.00`
- Checkpoints:
  `checkpoints/navigator-ablation-a1-bc1-r9/nnunet-actual/`
- Verdict: **failed**. One imitation epoch is insufficient, and the first PPO
  batch regressed held-out Dice. The expert's 18/18 score is not counted as
  learned progress.
- Next ablation: five-epoch scheduled DAgger warm-start on the identical split,
  followed by the same one-batch PPO probe.

## 2026-07-26 — A2: five-epoch full-roll-in DAgger plus one PPO batch

- Service: `navigator-ablation-a2-dagger5-r9.service`
- Split and geometry: identical to A1.
- DAgger epoch outcomes (`policy roll-in`, expert-completed training routes,
  supervised loss):
  - epoch 1: `0.00`, 18/18, `1.9426`
  - epoch 2: `0.25`, 12/18, `2.3267`
  - epoch 3: `0.50`, 2/18, `2.5960`
  - epoch 4: `0.75`, 0/18, `2.8825`
  - epoch 5: `1.00`, 0/18, `3.3163`
- Policy-only held-out results:
  - immediately after DAgger: mean Dice about `0.01`, traversal success `0.00`
  - after one PPO batch: mean Dice `0.0087`, traversal success `0.00`
- Verdict: **failed**. Raising policy roll-in to 100% faster than the learner
  can recover produces increasingly off-route states and degrades both the
  supervised objective and held-out Dice.
- Next ablation: repeat the same five epochs with pure expert rollouts
  (`behavior_cloning_max_policy_probability=0`) to separate representation
  fitting from recovery-state distribution shift.

## 2026-07-26 — A3: five-epoch pure-expert imitation plus one PPO batch

- Service: `navigator-ablation-a3-bc5-r9.service`
- Split and geometry: identical to A1/A2; policy roll-in fixed at zero.
- BC loss by epoch: `1.9426`, `1.6312`, `1.5506`, `1.5027`, `1.4651`.
- Generated expert completion stayed 18/18 in every epoch.
- Policy-only held-out results:
  - immediately after BC: mean Dice about `0.10`, traversal success `0.00`
  - after one PPO batch: mean Dice `0.14492`, endpoint reach `0.00`,
    traversal success `0.00`, mean endpoint distance `101.73 mm`
  - per-case final Dice: `0.04119`, `0.24865`
- Frozen BC checkpoint audit on two train and two held-out cases:
  - train `s0001`: Dice `0.13144`, endpoint not reached
  - train `s0038`: Dice `0.06568`, endpoint not reached
  - held-out `s0190`: Dice `0.10496`, endpoint not reached
  - held-out `s0918`: Dice `0.08595`, endpoint not reached
- Verdict: **failed representation fit**. The policy cannot reproduce even its
  own training demonstrations, so more PPO is not justified.
- Finding: the expert is derived from the nnU-Net small-bowel mask, but the
  actor never observes that mask. It receives current/previous CT, wall
  response, and cumulative path only, despite the environment using the
  segmentation to constrain every action.
- Next ablation: expose the local segmentation patch as a fifth observation
  channel and repeat pure-expert fitting before reintroducing DAgger or PPO.

## 2026-07-26 — A4: segmentation observation, five BC epochs, one PPO batch

- Service: `navigator-ablation-a4-segobs-bc5-r9.service`
- Change from A3: add the local nnU-Net small-bowel mask as observation channel
  five. Split, oracle, tube, BC schedule, and one-batch PPO probe are unchanged.
- Tests: full suite **31 passed** before training.
- BC loss by epoch: `1.8472`, `1.5036`, `1.4340`, `1.3956`, `1.3624`.
- Policy-only held-out results:
  - immediately after BC: mean Dice about `0.15`, traversal success `0.00`
  - after one PPO batch: mean Dice `0.33859`, endpoint reach `0.00`,
    traversal success `0.00`
  - per-case Dice: `0.17277`, `0.50441`
  - per-case endpoint distance: `95.62 mm`, `125.80 mm`
- Verdict: **first meaningful policy progress, not acceptance**. Mean held-out
  Dice more than doubled relative to A3 and one case cleared 0.40, but the mean
  remains below 0.40 and neither endpoint was reached.
- Decision: promote the exact final A4 checkpoint to a 100k-frame PPO
  continuation with held-out validation every 10,240 frames. Require Dice and
  endpoint metrics to improve jointly.

## 2026-07-26 — A5: A4 checkpoint plus standard PPO continuation

- Service: `navigator-ablation-a5-segobs-ppo100k-r9.service`
- Initialization: exact A4 final checkpoint.
- PPO: learning rate `5e-5`, four update epochs, validation/save every 10,240
  frames; planned 100k frames.
- Held-out trajectory:
  - 10,240: Dice `0.23596`, endpoint reach `0.00`, distance `76.97 mm`
  - 20,480: Dice `0.27462`, endpoint reach `0.00`, distance `48.33 mm`
  - 30,720: Dice `0.12992`, endpoint reach `0.00`, distance `133.86 mm`
  - 40,960: Dice `0.21992`, endpoint reach `0.00`, distance `153.84 mm`
- Verdict: **stopped early / unstable**. Dice and endpoint distance improved
  jointly through 20k, then both deteriorated sharply. Preserved best
  checkpoint: `checkpoint_20480best.pth`.
- Next ablation: restart from A4 and reduce PPO update epochs from four to one,
  keeping the learning rate unchanged. This isolates optimizer pressure before
  changing learning rate.

## 2026-07-26 — A6: A4 checkpoint plus one-update-epoch PPO

- Service: `navigator-ablation-a6-segobs-ppo1epoch-r9.service`
- Initialization: exact A4 final checkpoint.
- Change from A5: reduce PPO update epochs from four to one; retain learning
  rate `5e-5`, 1,024-frame batches, and the unchanged held-out split.
- Scheduled held-out validation trajectory:
  - 10,240: Dice `0.25452`, endpoint reach `0.00`, distance `115.18 mm`
  - 20,480: Dice `0.57074`, endpoint reach `0.00`, distance `127.67 mm`
  - 30,720: Dice `0.39692`, endpoint reach `0.00`, distance `129.55 mm`
  - 40,960: Dice `0.61105`, endpoint reach `0.00`, distance `175.23 mm`
  - 51,200: Dice `0.44272`, endpoint reach `0.00`, distance `133.04 mm`
- The 60,000-frame run completed successfully. Because the regular
  10,240-frame cadence did not evaluate the final frame, the exact final
  checkpoint was evaluated separately with the same deterministic mean policy:
  - `s0190`: Dice `0.46429`, endpoint reached, traversal success, 425 actions
  - `s0918`: Dice `0.41787`, endpoint reached, traversal success, 317 actions
  - mean Dice: **`0.44108`**
  - endpoint reach rate: **`1.00`**
  - joint traversal success rate at Dice >= 0.40: **`1.00`**
- Output:
  `results/navigator_nnunet/ablation-a6-final-eval/summary.json`
- Original verdict at run time: the configured 9 mm gate passed on both
  held-out cohort cases. The independent M0 audit below shows that this does
  **not** pass the original 6 mm construction, so it must not be reported as
  final learned acceptance.

## 2026-07-26 — A6.1: broader unseen connected-case audit

- Checkpoint: exact A6 final model.
- Cases: all eight cached cases with 26-neighbor-connected endpoints that were
  outside the complete 20-case A1–A6 cohort:
  `s0253`, `s0551`, `s0680`, `s0898`, `s0899`, `s0978`, `s1157`, `s1240`.
- These cases were not used for either BC or PPO. They are heterogeneous:
  segmentation volumes range from 474 to 312,869 voxels.
- Result:
  - mean Dice: `0.43444`
  - endpoint reach rate: `0.25`
  - joint traversal success rate: `0.25`
  - successful cases: `s0898` (Dice `0.62899`) and `s0978`
    (Dice `0.66617`)
- Output:
  `results/navigator_nnunet/ablation-a6-extra-unseen-eval/summary.json`
- Verdict: **coverage progress generalizes, endpoint traversal does not yet
  generalize reliably**. The preliminary two-case validation pass is real but
  must not be presented as robust end-to-end performance.

## 2026-07-26 — T6: endpoint-reward scaling regression suite

- Finding motivating the not-yet-run ablation: the bounded Dice potential has
  scale `50`, while the entire telescoping GDT endpoint potential had scale
  only `1`. A6 nevertheless reached the gate at its final checkpoint, so no
  stronger-reward training was started.
- Change prepared: expose a separate `gdt_reward_scale`, preserve the
  telescoping/no-loop-reward property, and include its full bound in the
  terminal failure penalty so failed episodes cannot end with positive return.
- Command: full pytest suite under `uv` on commander.
- Result: **32 passed** in 4.51 s; 18 upstream deprecation warnings.

## 2026-07-26 — A7: balanced Dice/GDT potential

- Service: `navigator-ablation-a7-segobs-gdt50-r9.service`
- Initialization and optimizer: same A4 checkpoint and one-update-epoch PPO
  schedule as A6.
- Change from A6: increase the *total bounded* GDT potential from `1` to `50`
  while retaining Dice potential `50`. Terminal failure becomes `-101`, which
  covers both full shaping bounds plus the invalid-movement scale. This remains
  non-exploitable by cycles because both shaping terms telescope.
- Scheduled held-out validation:
  - 10,240: Dice `0.15438`, endpoint distance `89.61 mm`
  - 20,480: Dice `0.19258`, endpoint distance `94.98 mm`
  - 30,720: Dice `0.17889`, endpoint distance `127.46 mm`
  - 40,960: Dice `0.14548`, endpoint distance `129.51 mm`
  - 51,200: Dice `0.13623`, endpoint distance `129.99 mm`
  - endpoint reach and joint traversal success remained `0.00` throughout.
- Exact final checkpoint: mean Dice `0.15334`, endpoint reach `0.00`, mean
  endpoint distance `151.41 mm`.
- Output:
  `results/navigator_nnunet/ablation-a7-final-eval/summary.json`
- Verdict: **failed**. Raising the GDT scale degraded both objectives rather
  than fixing endpoint generalization. The launcher default remains `1`; the
  separate scale stays configurable for controlled future studies.

## 2026-07-26 — A8: 20 additional pure-expert BC epochs

- Service: `navigator-ablation-a8-segobs-bc20-r9.service`
- Initialization: exact A4 final checkpoint.
- Change: 20 pure-expert behavior-cloning epochs
  (`policy_probability=0.00`) followed by a single 1,024-frame, one-update-epoch
  PPO probe.
- Demonstration health: every epoch completed 18/18 training routes; the
  supervised loss continued falling through the final epoch:
  - epoch 4: `1.2769`
  - epoch 10: `1.1625`
  - epoch 15: `1.0890`
  - epoch 20: `0.9811`
- Policy-only held-out results:
  - pure BC checkpoint: mean Dice about `0.08`, traversal success `0.00`
  - after one PPO batch: mean Dice `0.18840`, endpoint reach `0.00`,
    traversal success `0.00`, endpoint distance `148.63 mm`
- Verdict: **failed generalization despite improving supervised fit**. More
  teacher-forced epochs alone overfit route actions and do not fix accumulated
  policy errors or the partially observed branch-ordering problem.

## 2026-07-26 — M0: independent A6 metric audit

- Purpose: verify the claimed A6 result without calling the environment's
  coverage counters or success flag.
- Method:
  - load each saved trajectory and the raw small-bowel label independently;
  - rasterize every consecutive trajectory segment;
  - construct the path tube with SciPy's independent taxicab distance
    transform, matching the environment's L1 dilation geometry;
  - recompute binary Dice directly as
    `2 * |tube ∩ bowel| / (|tube| + |bowel|)`;
  - recompute Euclidean final-to-goal distance from saved coordinates.
- At the A6 run's configured L1 radius of 9 mm:
  - `s0190`: recomputed Dice `0.464288875` versus reported `0.464288861`
  - `s0918`: recomputed Dice `0.417873926` versus reported `0.417873919`
  - the counter implementation is numerically correct to floating-point
    precision.
- At the original L1 radius of 6 mm:
  - `s0190`: Dice `0.28415`
  - `s0918`: Dice `0.25862`
  - mean Dice: `0.27139`, below the requested `0.40`.
- Endpoint audit:
  - `s0190` stopped `8.746 mm` from the saved goal;
  - `s0918` stopped `8.617 mm` from the saved goal;
  - neither path reached the exact endpoint or the original 6 mm tolerance;
  - both passed only because the same radius parameter had been raised to
    9 mm and is also reused as the endpoint tolerance.
- Geometry cross-check: using a true Euclidean 9 mm tube instead of the
  implemented L1 tube gives Dice `0.58614` and `0.51846`; therefore the code's
  "radius" is an L1 diamond, not an isotropic physical sphere.
- Verdict: **retract the A6 acceptance claim under the original task
  definition**. The metric counters are internally correct, but evaluation
  geometry and endpoint tolerance are coupled. Raising one parameter both
  thickened the predicted path and relaxed what counted as end-to-end
  traversal. A fixed 6 mm tube and a separately specified endpoint tolerance
  are required for a trustworthy acceptance test.
