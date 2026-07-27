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
  traversal. At this audit stage a fixed 6 mm tube and a separately specified
  endpoint tolerance were proposed; M0.1 supersedes the radius choice after
  correcting the voxel/mm interpretation.

## 2026-07-26 — M0.1: anatomical-radius clarification

- Correction: the previous M0 conclusion treated 6 mm as the required
  anatomical path radius. At the actual 1.5 mm spacing, A6's 9 mm radius is six
  voxels and corresponds to an 18 mm-diameter construction, which is plausible
  for small bowel. A literal 6 mm radius is four voxels and produces only a
  12 mm-diameter construction.
- Revised decision: preregister the **9 mm radius** before further training.
  This is not selected per checkpoint or per case. The invalid part of A6 was
  coupling that radius to a 9 mm endpoint tolerance and implementing an L1
  diamond, not the anatomical radius itself.
- New fixed acceptance geometry:
  - Euclidean physical-space tube radius: `9 mm`;
  - independent endpoint tolerance: `3 mm`;
  - traversal success: Dice `>= 0.40` **and** endpoint distance `<= 3 mm`.
- Sensitivity analyses may report 6, 7.5, 9, and 10.5 mm radii, but checkpoint
  selection and the headline result use only the preregistered 9 mm radius.

## 2026-07-26 — G1/T7: trustworthy real-data traversal goal

- New goal: obtain at least `0.40` Dice and full start-to-end traversal on an
  untouched nnU-Net test cohort, using the M0.1 geometry.
- Cohort protocol:
  - preflight all 585 complete raw cases;
  - record every rejected case and reason;
  - freeze a seed-42 80/10/10 train/validation/test split after eligibility;
  - after the automated, model-independent eligibility pass and split, use test
    cases only once after architecture, reward, and checkpoint are fixed; never
    use test metrics for BC, PPO, ablations, or checkpoint choice.
- Training changes:
  - 2,048-step horizon;
  - coverage-then-end skeleton demonstrations;
  - 12-value route context: time, geodesic progress, Dice coverage, normalized
    global position, previous direction, and goal direction;
  - checkpoint ranking is lexicographic by traversal success, endpoint reach,
    Dice, then endpoint distance;
  - validation uses an independent SciPy Euclidean-distance implementation and
    aborts rather than silently skipping failed cases;
  - validation iterates the immutable manifest in a fixed order even though
    the training loader remains shuffled;
  - diagnostic checkpoints and validation outputs use fresh, run-specific
    directories, and every scheduled validation retains a step-numbered JSON
    metric snapshot in addition to TensorBoard.
- Metric and launcher regressions: `9 mm / 1.5 mm = 6` path-radius voxels,
  while `3 mm / 1.5 mm = 2` endpoint-tolerance voxels.
- Verification on commander with `uv`: **40 passed**, 18 upstream deprecation
  warnings.
- Preflight service: `navigator-preflight-v1.service`.
- Preflight output: `experiments/navigator-nnunet-v1/` (immutable).
- Preflight result:
  - 585 complete matched cases inspected;
  - 468 eligible cases with connected anatomical endpoints and an executable
    expert route;
  - 117 rejected cases retained with their exception type and message;
  - frozen seed-42 split: 374 train / 46 validation / 48 sealed test;
  - median expert-route length: 2,787 dense voxels; maximum: 9,689.
- Diagnostic protocol after preflight: one behavior-cloning warm-start epoch
  followed by 250k PPO steps. Scale to one million only if validation shows
  joint improvement rather than Dice-only or endpoint-only progress.

## 2026-07-26 — T7.1: full-cohort preflight corrections

- The first preflight attempt retained CuCIM/CuPy allocations across cases and
  reached 8.5 GiB of GPU memory by case 82. Added an explicit per-case CuPy
  memory-pool release and restarted; derived caches remain reusable and no
  immutable manifests had yet been written.
- The corrected run exposed an expert-route bug on `s0236`: endpoints were
  mask-connected, but the nearest skeleton node belonged to a nearby
  disconnected segmentation fragment. Restricting skeletonization to the
  26-connected component containing both anatomical endpoints fixes the
  attachment without dilating or repairing the segmentation.
- Added synthetic regressions for a thick connected target with a closer
  disconnected distractor and for an empty medial skeleton. Focused `uv`
  result: **5 passed**.
- Bumped the expert-route cache key from `skeleton_tree_v1` to
  `skeleton_tree_v2` so preflight cannot reuse routes generated before the
  connected-component correction.
- Case `s0422` exposed a connected but degenerate component for which Lee
  skeletonization is empty. Such cases now fall back to the exact
  mask-constrained endpoint route instead of being falsely labeled
  anatomically disconnected.
- Preflight restarted under the same `navigator-preflight-v1.service` name;
  output remains pending and versioned.

## 2026-07-27 — G1.1: pre-metric anatomical eligibility correction

- The completed v1 preflight found 468 endpoint-connected cases, but its
  model-independent metadata exposed an anatomically invalid lower tail:
  expert routes ranged from 1 to 9,689 dense voxels, and `s0422` had coincident
  endpoints, a one-voxel route, and only 31 traversable segmentation voxels.
- This was discovered before running the validation oracle or inspecting any
  learned validation/test metric. The v1 manifests remain immutable and
  retained for audit.
- Preflight v2 adds a fixed minimum **300 mm physical expert-route length**.
  This is deliberately far below an adult small-bowel length, retains
  thin-but-long segmentations, and rejects tiny fragments or coincident
  endpoints that would make traversal accuracy artificially easy.
- The physical route length and segmentation volume are recorded per eligible
  case. The seed-42 80/10/10 split is regenerated only after this
  model-independent filter. The v2 test split remains sealed.
- Exact-source verification on commander:
  `PYTHONPATH=$PWD/src uv run --no-sync pytest -q tests` — **41 passed** with
  18 upstream deprecation warnings.
- Preflight v2 result:
  - 585 complete matched cases inspected;
  - 413 eligible;
  - 117 rejected for disconnected anatomical endpoints;
  - 55 rejected for an expert route shorter than 300 mm;
  - frozen split: 330 train / 41 validation / 42 sealed test.

## 2026-07-27 — A8: v2 validation oracle and action-magnitude diagnostic

- Cohort: all 41 frozen v2 validation cases; no test cases inspected.
- Configuration: skeleton-covering oracle, Euclidean 9 mm path tube,
  independent 3 mm endpoint tolerance, Dice threshold 0.40, 2,048 actions.
- Result:
  - mean Dice: `0.68253`; minimum Dice: `0.50573`;
  - endpoint reach and joint traversal success: `39/41` (`0.95122`);
  - failures: `s0383` ended 20.07 mm from the endpoint and `s1031` ended
    4.50 mm away, both at the 2,048-step ceiling.
- A 4,096-step rerun on only those two cases stopped at exactly the same final
  coordinates and Dice values. Saved path tails showed deterministic two-point
  cycles, proving this was not a horizon shortage.
- Cause: `_project_action_to_allowed_displacement` normalized every nonzero
  action to the maximum step length. The expert therefore could not request a
  shorter final move and repeatedly overshot its waypoint.
- Fix under test: preserve continuous action magnitude, shorten only invalid
  rays, and encode expert waypoint length as well as direction. The registered
  metric geometry is unchanged.
- The magnitude fix resolved `s1031` in 886 actions at the exact endpoint
  (Dice `0.65167`). `s0383` still cycled at an intermediate bend.
- A deterministic cursor trace found the remaining cause: the expert advanced
  its route index whenever a future waypoint was within four Euclidean voxels.
  At folded bowel contacts this skipped up to 13 ordered waypoints at once,
  eventually targeting a point across a mask barrier.
- Follow-up fix under test: look ahead by at most `max_step_vox` contiguous
  route indices and advance only to an exactly traversable waypoint. This
  retains a monotonic topological cursor and still permits safe straight-line
  compression of dense routes.
- Two-case regression after both fixes:
  - `s0383`: Dice `0.71052`, endpoint distance `1.50 mm`, success in 918 steps;
  - `s1031`: Dice `0.66219`, endpoint distance `2.12 mm`, success in 939 steps.
- Complete corrected oracle rerun on all 41 frozen validation cases:
  - mean Dice: `0.68318`;
  - minimum Dice: `0.50510`;
  - endpoint reach rate: `1.00`;
  - traversal success rate: `1.00`;
  - average endpoint distance: `0.635 mm`;
  - maximum action count: 1,476 / 2,048.
- Verdict: **oracle ceiling accepted**. The 2,048-step horizon is sufficient;
  training may proceed without changing the preregistered metric or horizon.

## 2026-07-27 — G2: v2 250k diagnostic host OOM

- Service: `navigator-g1-diagnostic-250k-v2.service`.
- Behavior cloning remained numerically healthy through 83/330 cases:
  teacher-forced traversal was 83/83, loss was approximately 1.45, and CUDA
  allocation stayed near 2.1 GiB.
- The kernel OOM killer terminated the service after 95 seconds. Unit peak:
  23.7 GiB resident memory and 26.8 GiB swap; this was not a CUDA OOM.
- Cause: `make_sb_env` wrapped the PyTorch DataLoader in `itertools.cycle`.
  `cycle` retains every yielded subject so it can replay them, which cached one
  complete multi-volume 3-D case per BC iteration and produced linear memory
  growth.
- Fix: replace `cycle(loader)` with a non-caching generator that repeatedly
  executes `yield from loader`. Add a regression for epoch repetition and
  restart into fresh v3 checkpoint/validation directories.

## 2026-07-27 — G3: v3 behavior-cloning checkpoint

- Service: `navigator-g1-diagnostic-250k-v3.service`.
- Exact-source verification before launch: **43 passed**, 18 upstream
  deprecation warnings.
- BC completed in 4 min 57 s:
  - 248,760 teacher-forced environment steps over 330 training cases;
  - mean loss `1.3019`;
  - teacher traversal success `326/330`;
  - checkpoint:
    `checkpoints/navigator-g1-diagnostic-250k-v3/nnunet-actual/behavior_cloning_model.pth`.
- Deterministic policy-mean evaluation on all 41 frozen validation cases:
  - mean Dice: `0.39918`;
  - endpoint reach: `1/41` (`0.02439`);
  - joint traversal success: `1/41` (`0.02439`);
  - average endpoint distance: `79.62 mm`;
  - average episode length: 2,011.7 / 2,048.
- Interpretation: one-pass BC nearly reaches the Dice target but does not
  reproduce long routes under closed-loop rollout. PPO must improve endpoint
  completion without selecting a Dice-only checkpoint.
- First PPO validation at 25,600 collected frames:
  - mean Dice: `0.46203`;
  - endpoint reach and traversal success: `3/41` (`0.07317`);
  - average endpoint distance: `93.57 mm`.
- Dice and the lexicographically primary success count improved over BC, but
  endpoint distance worsened. Continue the diagnostic; do not accept this
  checkpoint as reasonable traversal accuracy.
- Subsequent validation:
  - 51,200: Dice `0.62355`, success `2/41`, endpoint distance `94.13 mm`;
  - 76,800: Dice `0.49241`, success `1/41`, endpoint distance `97.71 mm`.
- The success trend was 3 → 2 → 1 while PPO increasingly optimized coverage.
  The service was stopped after 76,800 steps; `checkpoint_25600best.pth`
  remains the lexicographically best artifact, but 3/41 is not reasonable
  accuracy.

## 2026-07-27 — G4: closed-loop BC/DAgger ablation

- Motivation: teacher-forced BC solved 326/330 training routes, while its
  deterministic held-out policy accumulated errors and completed only 1/41.
  More pure-expert epochs previously overfit without correcting visited-state
  drift.
- One-factor change from G3: three BC epochs with learned-policy rollout
  probability `0.00`, `0.25`, then `0.50`; PPO settings and fixed metric remain
  unchanged.
- Correctness fix: after a learned-policy action, re-synchronize the expert
  cursor only within a bounded forward route window. Do not advance the cursor
  as if the unexecuted expert action had occurred, and do not match a distant
  spatially touching loop.
- Planned service: `navigator-g1-dagger-bc3-100k-v1.service`.
- Epoch results:
  - epoch 1, policy probability 0.00: loss `1.3019`, success `326/330`;
  - epoch 2, policy probability 0.25: loss `1.2597`, success `325/330`;
  - epoch 3, policy probability 0.50: loss `1.4425`, success `316/330`.
- Deterministic frozen-validation result after epoch 3:
  - mean Dice: `0.33844`;
  - endpoint reach and traversal success: `0/41`;
  - average endpoint distance: `99.40 mm`;
  - all episodes exhausted 2,048 steps.
- Verdict: **failed**. The recovery labels kept mixed training rollouts
  healthy but degraded held-out policy behavior. Stop before PPO.

## 2026-07-27 — G5: moderate endpoint-potential ablation

- Initialization: exact G3 one-epoch BC checkpoint, before PPO.
- One-factor change from G3 PPO: increase the total bounded GDT endpoint
  potential from `1` to `5`; retain Dice potential `50`, all architecture and
  optimizer settings, manifests, metric geometry, and horizon.
- This is intentionally far below the previously failed A7 scale `50`.
  Terminal failure remains strictly negative because its bound automatically
  includes the larger GDT potential.
- Planned service: `navigator-g1-gdt5-100k-v1.service`.
- Validation at 25,600 steps:
  - mean Dice: `0.39771`;
  - endpoint reach and traversal success: `1/41`;
  - average endpoint distance: `86.52 mm`.
- Compared with G3 at the same step, mean endpoint distance is modestly
  better, but Dice and the primary completed-traversal count are worse
  (`1/41` versus `3/41`). Continue to one more checkpoint only to test slower
  convergence; do not select this checkpoint.
- Validation at 51,200:
  - mean Dice: `0.46020`;
  - endpoint reach and traversal success: `3/41`;
  - average endpoint distance: `79.94 mm`.
- Verdict: **tie / not selected**. G5 eventually ties G3's best completed
  traversal count and improves mean endpoint distance, but its Dice is
  slightly lower (`0.46020` versus `0.46203`). Under the fixed lexicographic
  rank, G3 remains best. Stop at the predeclared second comparison.

## 2026-07-27 — G6: moderate two-epoch DAgger probe

- Motivation: G4 epoch 2 retained `325/330` mixed-rollout training successes
  with loss `1.2597`; the more aggressive 50%-policy epoch 3 worsened held-out
  behavior.
- One-factor change from G3 BC: add one second epoch with 25% learned-policy
  rollout actions and bounded route-cursor recovery. Do not run the degraded
  50% epoch.
- PPO probe is limited to 1,024 steps; the policy-only validation emitted
  immediately after BC is the decision metric.
- Resource protocol change requested before launch: iterative validation uses
  a fixed three-case sentinel set selected only by model-independent expert
  route length:
  - `s1389`: shortest route, 345.19 mm;
  - `s0224`: median route, 5,564.12 mm;
  - `s0120`: longest route, 11,094.49 mm.
- The three-case smoke metric is diagnostic, not a headline accuracy claim.
  Run the complete 41-case frozen validation cohort only after a configuration
  clearly passes this smoke gate.
- Planned service: `navigator-g1-dagger-bc2-probe-v1.service`.
- Epoch results:
  - epoch 1, policy probability 0.00: loss `1.3019`, success `326/330`;
  - epoch 2, policy probability 0.25: loss `1.2597`, success `325/330`.
- Three-case mean-action smoke result:
  - mean Dice: `0.55433`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `86.29 mm`;
  - all episodes exhausted 2,048 steps.
- Matched G3 one-epoch BC baseline on the same three cases was Dice `0.42551`,
  success `0/3`, and endpoint distance `71.81 mm`.
- Verdict: **failed**. The second DAgger epoch improves overlap but worsens
  endpoint localization and does not complete a traversal. Do not expand to
  the 41-case validation set.

## 2026-07-27 — G7: paper-style Beta-mode evaluation

- Motivation: the original method specifies a deterministic Beta mode, while
  the corrected BC implementation supervises and evaluates the distribution
  mean.
- One-factor change from G6: evaluate the exact same pre-PPO BC checkpoint
  with the Beta mode. Dataset, weights, geometry, reward, horizon, and the
  fixed three validation subjects remain unchanged.
- Three-case result:
  - mean Dice: `0.44344`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `42.78 mm`.
- Verdict: **promising diagnostic, not a pass**. Mode more than halves the G6
  mean-action endpoint error, but no case reaches the endpoint and the longest
  case has only `0.13039` Dice. Do not expand validation.

## 2026-07-27 — G8: mode-aligned one-epoch BC

- One-factor change from G3: use the differentiable Beta mode for BC action
  regression, BC rollouts, and deterministic validation. Retain a single pure
  expert epoch, all rewards, geometry, architecture, and the same three-case
  smoke cohort.
- Planned service: `navigator-g1-mode-bc1-probe-v1.service`.
- Epoch result: loss `1.2114`, teacher traversal success `326/330`.
- Three-case result:
  - mean Dice: `0.34085`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `90.76 mm`;
  - all episodes exhausted 2,048 steps.
- Verdict: **failed**. Directly regressing the sharper mode harms both overlap
  and endpoint localization. The G7 benefit came from reading the mode of a
  mean-trained distribution, not from fitting the mode itself.

## 2026-07-27 — G9: PPO from mean-BC with mode selection

- Initialization: exact G6 two-epoch mean/DAgger BC checkpoint, before its
  1,024-step PPO probe.
- One-factor change from G7: continue PPO optimization while retaining
  deterministic Beta-mode validation and the fixed three-case smoke cohort.
- Budget: 100,000 collected frames, with validation every 25 iterations
  (25,600 frames). Stop early only for a clear regression; rank checkpoints by
  traversal success, endpoint reach, Dice, then endpoint distance.
- Planned service: `navigator-g1-mode-ppo-100k-v1.service`.
- Validation at 25,600 frames:
  - mean Dice: `0.66081`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `105.06 mm`.
- Validation at 51,200 frames:
  - mean Dice: `0.72538`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `108.13 mm`.
- Verdict: **failed and stopped early**. PPO consistently increased coverage
  while moving farther from the endpoint. This is the same reward imbalance
  observed in G3; do not spend the remaining 48,800-frame budget.

## 2026-07-27 — G10: corrected-policy balanced Dice/GDT retry

- Initialization, validation statistic, cohort, optimizer, and geometry are
  identical to G9.
- One-factor change: increase the total bounded GDT endpoint potential from
  `1` to `50`, matching the Dice potential. This retries A7 only because the
  current action-magnitude projection and G6 checkpoint did not exist then.
- The shaping remains a telescoping potential, and terminal failure includes
  both full potential bounds, so cycles and incomplete high-return episodes
  remain impossible by construction.
- Budget: 51,200 frames with comparisons at 25,600 and 51,200.
- Planned service: `navigator-g1-mode-gdt50-51k-v1.service`.
- Validation at 25,600 frames:
  - mean Dice: `0.35430`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `71.13 mm`;
  - two cases moved close to their endpoint (`19.96` and `7.50 mm`), while
    the median-length case regressed to `185.94 mm`.
- Validation at 51,200 frames:
  - mean Dice: `0.38771`;
  - endpoint reach and traversal success: `1/3`;
  - average endpoint distance: `29.38 mm`;
  - `s1389` completed end to end in 1,475 steps with Dice `0.62602` and
    endpoint error `2.12 mm`;
  - `s0224` reached Dice `0.35933`, endpoint error `69.37 mm`;
  - `s0120` reached Dice `0.17778`, endpoint error `16.64 mm`.
- Verdict: **first genuine smoke traversal progress**. The mean Dice narrowly
  misses 0.40 and two cases remain incomplete, so this is not yet reasonable
  accuracy and must not be expanded to the full validation cohort.

## 2026-07-27 — G11: continuous balanced-GDT 200k schedule

- Repeat G10 from the identical G6 BC checkpoint with the same seed and all
  settings, but extend the continuous PPO/optimizer schedule to 200,000
  frames. This avoids treating a new optimizer loaded at 51,200 as an exact
  continuation.
- Validation remains the same fixed three cases every 25,600 frames. Preserve
  G10 independently; stop G11 only after a sustained success regression or
  completion of the schedule.
- Planned service: `navigator-g1-mode-gdt50-200k-v1.service`.
- The nominally identical repeat did not reproduce G10 exactly:
  - 25,600: Dice `0.30519`, success `0/3`, endpoint error `87.34 mm`;
  - 51,200: Dice `0.37683`, success `0/3`, endpoint error `26.68 mm`;
  - 76,800: Dice `0.41118`, success `0/3`, endpoint error `76.04 mm`;
  - later checkpoints increased Dice, peaking at `0.52233`, but success
    remained `0/3` throughout.
- Final 200,000-frame result: Dice `0.47251`, success `0/3`, endpoint error
  `26.32 mm`.
- Verdict: **failed robustness check**. Balanced GDT reliably brings some
  endpoints closer and can exceed the Dice threshold, but the G10 completion
  is not stable under a nominal repeat. Preserve G10; do not report a robust
  success rate from these three cases.

## 2026-07-27 — G12: 48 mm anatomical-context BC

- Motivation: the 24 mm observation is approximately one bowel diameter and
  often cannot show a complete fold or nearby branch. This leaves a feedforward
  policy locally aliased even though it sees the segmentation channel.
- One-factor change from one-epoch mean BC: double the actor patch to 48 mm.
  Keep the expert route hidden, train the Beta mean, evaluate the Beta mode,
  and retain all geometry and the fixed three-case validation cohort.
- Use BC batch size 16 and PPO batch size 16 for GPU memory safety; the
  post-BC PPO probe is only 256 frames and is not the decision metric.
- Planned service: `navigator-g1-patch48-bc1-probe-v1.service`.
- Epoch result: loss `1.2649`, teacher traversal success `326/330`.
- Three-case mode result:
  - mean Dice: `0.54888`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `70.92 mm`.
- Verdict: **better representation, incomplete traversal**. Compared with the
  matched 24 mm one-epoch BC baseline, the larger patch improves smoke Dice
  but does not solve endpoint localization by itself.

## 2026-07-27 — G13: 48 mm BC plus balanced-GDT PPO

- Initialization: exact G12 pre-PPO BC checkpoint.
- Continue with the G10 balanced GDT/Dice potentials (`50` each), Beta-mode
  validation, and otherwise unchanged geometry.
- Memory-bounded schedule: 256 frames per batch, PPO minibatch 16, 51,200
  total frames, validation every 25,600 frames on the same three cases.
- Planned service: `navigator-g1-patch48-gdt50-51k-v1.service`.
- Validation at 25,600 frames: Dice `0.60588`, success `0/3`, endpoint error
  `70.05 mm`.
- Validation at 51,200 frames: Dice `0.45145`, success `0/3`, endpoint error
  `50.39 mm`.
- Verdict: **failed**. Larger local context preserves coverage but does not
  resolve global route aliasing.

## 2026-07-27 — G14: planner-guided local goal-distance observation

- This is explicitly not a pure local VoxTrack/PPO result. Add one actor
  channel containing the local mask-constrained distance-to-endpoint potential,
  expressed as clipped progress relative to the current voxel.
- The channel is computed only from the same segmentation and requested
  endpoint available at inference. It contains no expert route, action, reward
  return, validation outcome, or sealed-test information.
- One-factor comparison uses the original 24 mm patch, one mean-BC epoch,
  Beta-mode validation, and the fixed three-case smoke cohort.
- Planned service: `navigator-g1-goaldist-bc1-probe-v1.service`.
- Epoch result: loss `1.3115`, teacher traversal success `326/330`.
- Three-case mode result:
  - mean Dice: `0.45021`;
  - endpoint reach and traversal success: `0/3`;
  - average endpoint distance: `16.11 mm`;
  - per-case endpoint error: `27.21`, `9.60`, and `11.52 mm`.
- Verdict: **strongest pre-PPO endpoint localization, not yet complete**. The
  explicit global potential resolves much of the aliasing but does not by
  itself satisfy the strict 3 mm endpoint tolerance.

## 2026-07-27 — G15: planner-guided BC plus balanced-GDT PPO

- Initialization: exact G14 pre-PPO checkpoint.
- Continue for 51,200 frames with equal bounded Dice/GDT potentials, 256-frame
  batches, minibatch 16, and validation only at 25,600 and 51,200 frames on
  the fixed three cases.
- Planned service: `navigator-g1-goaldist-gdt50-51k-v1.service`.
- Validation at 25,600 frames:
  - Dice `0.25779`;
  - endpoint reach and traversal success `1/3`;
  - endpoint error `4.37 mm`.
- Validation at 51,200 frames:
  - Dice `0.26416`;
  - endpoint reach `3/3`;
  - traversal success `1/3`;
  - endpoint error `1.71 mm`.
- The two incomplete cases have Dice `0.15364` and `0.10474` despite reaching
  their endpoints. Verdict: **endpoint solved, coverage collapsed**. The
  observable goal potential plus scale 50 over-optimizes the mask-shortest
  route and does not satisfy the joint objective.

## 2026-07-27 — G16: planner-guided moderate-GDT PPO

- One-factor change from G15: reduce the total GDT potential from `50` to `5`;
  retain the Dice potential `50`, exact G14 BC checkpoint, optimizer, schedule,
  geometry, mode validation, and fixed three cases.
- Planned service: `navigator-g1-goaldist-gdt5-51k-v1.service`.
- Original 25,600-frame metric: Dice `0.40656`, reported success `0/3`,
  endpoint error `5.35 mm`. Audit found `s1389` terminated after 180 steps
  with Dice `0.75070`, but independent physical metrics rejected its
  `3.000000238 mm` endpoint distance against a `3.0 mm` threshold.
- Correctness fix: physical-radius and endpoint-threshold comparisons now use
  an eight-float32-epsilon, scale-aware boundary tolerance. A regression test
  proves float32 spacing roundoff cannot reject a mathematically exact
  boundary; genuinely more distant endpoints remain unchanged.
- Evaluation-only rerun of the exact 25,600 checkpoint after the metric fix:
  - mean Dice `0.40623`;
  - endpoint reach and traversal success `1/3`;
  - endpoint error `5.35 mm`.
- Original 51,200-frame result: Dice `0.34067`, endpoint reach `2/3`,
  traversal success `1/3`, endpoint error `2.87 mm`.
- Verdict: **best joint mean Dice so far, but only 1/3 complete**. Scale 5
  preserves substantially more coverage than scale 50.

## 2026-07-27 — G17: one additional planner-guided BC epoch

- Initialization: exact G14 one-epoch BC checkpoint.
- One additional pure-expert mean-supervision epoch with a fresh AdamW state;
  no DAgger roll-in. Evaluate Beta mode on the fixed three cases before the
  256-frame non-decision PPO probe.
- Planned service: `navigator-g1-goaldist-bc-plus1-probe-v1.service`.
- Additional epoch result: loss `1.0903`, teacher success `326/330`.
- Three-case result: Dice `0.36127`, endpoint error `10.92 mm`, success `0/3`.
- Verdict: **failed / overfit**. Lower supervised loss does not improve
  closed-loop held-out traversal.

## 2026-07-27 — G18: combined 48 mm anatomy and goal-distance BC

- Combine the two independently useful state improvements: the 48 mm local
  anatomical patch from G12 and the optional goal-distance channel from G14.
- Train one mean-BC epoch from scratch with batch 16; evaluate Beta mode on the
  fixed three cases before a 256-frame non-decision PPO probe.
- This remains a planner-guided result because the goal-distance channel is
  present.
- Planned service: `navigator-g1-patch48-goaldist-bc1-probe-v1.service`.
- Epoch result: loss `1.2680`, teacher success `326/330`.
- Three-case result:
  - mean Dice `0.63343`;
  - per-case Dice `0.67304`, `0.57256`, `0.65470`;
  - endpoint reach and traversal success `0/3`;
  - endpoint error `69.55 mm`.
- Verdict: **strongest and most consistent coverage representation**, but BC
  alone still does not localize endpoints.

## 2026-07-27 — G19: combined model plus moderate-GDT PPO

- Initialization: exact G18 pre-PPO checkpoint.
- Continue with GDT potential `5`, Dice potential `50`, 256-frame batches,
  minibatch 16, and evaluations at 25,600 and 51,200 frames on three cases.
- Planned service: `navigator-g1-patch48-goaldist-gdt5-51k-v1.service`.
- At 25,600 frames: Dice `0.31933`, endpoint reach and traversal success
  `1/3`, endpoint error `6.16 mm`.
- At 51,200 frames: Dice `0.24810`, endpoint reach and traversal success
  `0/3`, endpoint error `10.90 mm`.
- Verdict: **failed**. Standard learning rate destroys the excellent G18
  coverage within one evaluation interval.

## 2026-07-27 — G20: low-rate combined-model PPO

- One-factor change from G19: reduce PPO learning rate from `5e-5` to `1e-5`.
  Initialize again from untouched G18 BC weights.
- Budget 128,000 frames, validation every 25,600 frames on three cases. Stop
  early only after a sustained regression or select a checkpoint only by the
  preregistered traversal-first rank.
- Planned service: `navigator-g1-patch48-goaldist-gdt5-lr1e5-128k-v1.service`.
- Results:
  - 25,600: Dice `0.40829`, endpoint reach and success `1/3`, endpoint error
    `5.52 mm`;
  - 51,200: Dice about `0.38`, success `0/3`;
  - 76,800: Dice `0.40276`, success `1/3`, endpoint error `5.74 mm`;
  - 102,400: Dice `0.39157`, success `0/3`;
  - final 128,000: Dice `0.39514`, success `0/3`, endpoint error `9.11 mm`.
- Verdict: **failed to improve completion count**. Lower rate slows but does
  not remove PPO's closed-loop instability; the first checkpoint remains best.

## 2026-07-27 — G21: coverage-gated hybrid endpoint planner

- Explicit planner-guided controller, not a pure RL result:
  - use the untouched G18 high-coverage BC policy;
  - once online path Dice reaches the fixed 0.40 success threshold, latch a
    local mask-constrained GDT descent until the requested endpoint;
  - never read the expert route or validation/test outcomes.
- This tests the mechanistic conclusion from G18–G20: learned local coverage
  and deterministic endpoint planning are individually strong, while PPO
  fails to learn a stable switch.
- Planned evaluation-only service:
  `navigator-g1-patch48-goaldist-hybrid-eval-v1.service`.
- Reproducible launcher:
  `scripts/run_navigator_g1_patch48_goaldist_hybrid_eval.sh`.
- Corrected independent three-case result:
  - mean Dice `0.43193`;
  - per-case Dice `0.47552`, `0.41094`, `0.40934`;
  - endpoint reach and traversal success `3/3`;
  - final endpoint error `0 mm` for all cases;
  - episode lengths `18`, `262`, and `360` steps.
- Verdict: **passes the requested three-case numeric smoke gate**. This is the
  first configuration to exceed mean Dice 0.40 and complete every selected
  end-to-end traversal.
- Interpretation boundary: this result demonstrates a viable hybrid system,
  not that PPO learned robust bowel traversal. The switch uses the online Dice
  gate and a deterministic distance planner derived from the segmentation and
  endpoint. It is transparent and available at inference, but must be reported
  separately from pure local-policy results.
- Per the resource constraint, do not run the full 41-case validation cohort
  yet. The fixed three cases were shortest/median/longest by expert-route
  length and are an iteration smoke set, not a final generalization estimate.
