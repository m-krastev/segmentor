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

## 2026-07-27 — M1: shared-encoder recurrent PPO feasibility

- Goal: establish executable GRU and state-space memory baselines before
  changing the observation and reward to the annotation-free formulation.
- Implementation:
  - one shared 3-D visual encoder feeding separate Beta actor and scalar critic
    heads through TorchRL's actor-value operator;
  - native TorchRL GRU with recurrent state registered in the environment;
  - dependency-free diagonal S5-style MIMO state-space recurrence with stable
    complex diagonal dynamics and bilinear discretization;
  - contiguous PPO sequence minibatches instead of shuffled transitions;
  - recurrent GAE with `vmap` disabled because reset flags introduce
    data-dependent control flow;
  - configurable memory width, S5 state size, sequence length, and backend.
- CUDA model and PPO-backward tests pass for both GRU and S5 configurations,
  including verification that actor and critic truly share parameters.
- End-to-end collector/training smoke on RTX 5070 Ti, three fixed cases used
  only to provide real environment tensors:
  - GRU: 64 frames, sequence 16, hidden 64, `62.1` final steps/s;
  - S5-style: same settings, `62.4` final steps/s;
  - both completed collection, GAE, PPO update, and checkpoint saving without
    OOM or recurrent-state errors.
- Matched production-capacity smoke with hidden/state width 256, sequence
  length 64, and a 256-frame PPO batch:
  - GRU: `154.9` final steps/s, `1.44 GiB` peak CUDA allocated and
    `2.25 GiB` reserved;
  - S5-style: `139.7` final steps/s, `1.56 GiB` peak CUDA allocated and
    `3.35 GiB` reserved;
  - each model has 602,967 unique trainable parameters and leaves substantial
    headroom on the 15.5 GiB GPU.
- Decision: use GRU as the primary recurrent baseline. It is faster, uses less
  CUDA memory, and relies on TorchRL's native recurrent module. Keep the
  dependency-free S5-style implementation as the state-space ablation; its
  sequential complex scan is deliberately correctness-oriented rather than a
  fused high-throughput implementation.
- Interpretation boundary: this is an execution/correctness result, not a
  learning result. The current environment still exposes segmentation-derived
  channels and rewards. No long recurrent training should be interpreted as an
  annotation-free result until those inputs and objectives are replaced.

## 2026-07-27 — M2: GRU PPO to 500k frames

- Goal: test whether the primary recurrent baseline begins learning useful
  closed-loop traversal by approximately 500k environment frames.
- Initialization: from scratch; no behavior cloning and no feed-forward
  checkpoint transfer.
- Configuration:
  - shared encoder plus one-layer GRU, hidden width 256;
  - recurrent sequence length 64;
  - 1,024 frames per collector batch, PPO minibatch 256, four update epochs;
  - learning rate `5e-5`, BF16 AMP;
  - 500,000 requested frames (the collector may finish the final complete
    batch slightly above this);
  - immutable v2 preflight training manifest with 330 connected,
    physically plausible subjects;
  - the fixed three-case v2 validation smoke manifest, keeping validation
    within the requested resource limit;
  - intermediate checkpoints approximately every 50k frames and one
    validation at the final approximately-500k boundary.
- Planned service: `navigator-gru-nnunet-500k-v1.service`.
- Checkpoints and TensorBoard events:
  `/home/matey/project/segmentor/checkpoints/navigator-gru-nnunet-500k-v1`
  (inside the directory already watched by `navigator-tensorboard.service`).
- Interpretation boundary: the current environment still uses
  segmentation-derived observations/rewards. This experiment assesses the
  recurrent PPO implementation, not the future annotation-free formulation.
- Launch note: an initial broad-dataset launch was rejected before its first
  PPO update when `s0628` was found to have disconnected anatomical
  endpoints. This confirmed that raw file completeness is not sufficient for
  training eligibility. The corrected launch uses the immutable v2 manifests.
- Result: **invalid for the annotation-free research question and stopped**.
  The service was terminated at 345,088 collected frames, before held-out
  validation. The last resumable checkpoint is 301,056 frames.
- Contamination audit:
  - the policy directly observed the GT small-bowel segmentation patch,
    current GT Dice, segmentation-derived geodesic progress, and goal
    direction;
  - action projection was constrained by the GT segmentation;
  - reward used GT coverage, segmentation-derived GDT, and an
    out-of-segmentation penalty;
  - termination used GT Dice and a label-derived endpoint.
- Training Dice from this run is not publishable evidence and must not be
  compared with annotation-free methods. Retain its checkpoints only as
  explicitly labeled privileged-debug artifacts.

## 2026-07-27 — M3: enforceable annotation-free environment

- Goal: make privileged-data access structurally impossible in the policy
  training path rather than relying on convention or zero reward weights.
- Contract:
  - policy dataset loader accepts CT, an image-derived Meijering cache, and one
    externally supplied start seed only;
  - no segmentation, endpoint, GDT, local peaks, or expert path is loaded into
    the training environment;
  - four spatial channels: previous CT patch, current CT patch,
    image-derived wall response, and agent-owned cumulative path;
  - seven context values: elapsed-time fraction, normalized position, and
    previous direction;
  - action projection is image-bounds-only;
  - training reward uses only agent-owned novelty, fixed step cost,
    image-derived wall response, and curvature;
  - episodes terminate only at the fixed horizon;
  - GT Dice/endpoint metrics are loaded by a separate evaluator only after a
    held-out rollout is complete;
  - held-out labels are report-only and cannot select a checkpoint or alter
    optimization; annotation-free runs save fixed-step checkpoints only.
- Configuration fails closed when annotation-free mode is combined with goal
  channels, coverage/GDT reward, endpoint termination, behavior cloning,
  expert path generation, or any other privileged option.
- Leakage regression: two environments with identical CT/seed but adversarially
  different masks, endpoints, GDTs, and expert paths produce identical
  observations, contexts, rewards, positions, and termination.
- Loader regression: annotation-free nnU-Net loading succeeds when label
  directories do not exist.
- Verification:
  - focused suite: 28 tests plus two subtests passed;
  - full suite: 55 tests plus two subtests passed;
  - 64-frame GRU CUDA smoke completed at 112 final steps/s with 914.6 MiB peak
    allocated and 1,316 MiB reserved.
- The CUDA smoke used label-derived seed coordinates as explicitly
  non-scientific temporary fixtures. A long clean run is blocked until an
  operator-supplied or image-derived seed manifest with honest provenance is
  available.
- Reward-quality warning: sigma-1 Meijering responses overlap heavily between
  bowel and background on the three audited cases. Passing the leakage gate
  does not establish that this intrinsic objective identifies bowel, and a
  long run must not be launched merely because the implementation executes.

## 2026-07-27 — M4: inference-clean policy with reward supervision

- Motivation:
  - remove the redundant previous-CT channel now that the policy has GRU
    memory and explicit previous direction;
  - replace the single sigma-1 Meijering response with a compact physically
    scaled filter bank;
  - separate policy observability from reward supervision.
- Policy inputs:
  - current CT patch;
  - dark-ridge and bright-ridge multiscale Meijering responses;
  - multiscale Gaussian band-pass magnitude;
  - smoothed gradient magnitude;
  - the agent-owned cumulative path mask.
- Filter scales are `3`, `6`, and `9 mm`, converted to voxels from image
  spacing. Full-volume filter maps are versioned and cached before reuse.
- Context remains seven non-label values: time, normalized position, and
  previous direction. There is no segmentation patch, Dice/progress scalar,
  goal direction, or goal-distance patch.
- Added an explicit reward-supervised/inference-clean protocol:
  - signed Dice-potential difference and signed GDT-potential difference are
    available during training;
  - GT can change reward and terminal scoring but cannot change policy inputs
    or bounds-only action dynamics;
  - behavior cloning, goal priors/channels, the hybrid planner, and expert
    paths are prohibited;
  - deployment requires CT, cached image filters, path history, and one start
    seed, but no mask or endpoint.
- This is a supervised-RL baseline, **not annotation-free training**. It must
  be named accordingly in every result.
- Separation regression: adversarially changing masks/GDT/endpoints changes
  reward while leaving observations, context, executed position, and action
  projection unchanged.
- Full suite: 56 tests plus two subtests passed.
- CUDA smoke, GRU hidden 64, 64 frames:
  - cold filter caches: 6.7 final steps/s;
  - warm filter caches: 59.9 final steps/s;
  - peak CUDA memory: 2,449 MiB allocated, 3,618 MiB reserved.
- Three-case filter audit against balanced negatives within 30 mm of the bowel
  mask, reported as polarity-independent AUC:
  - `s0120`: CT 0.733, dark Meijering 0.686, bright Meijering 0.590,
    band-pass 0.526, gradient 0.614;
  - `s0224`: CT 0.534, dark Meijering 0.585, bright Meijering 0.559,
    band-pass 0.671, gradient 0.711;
  - `s1389`: CT 0.678, dark Meijering 0.797, bright Meijering 0.742,
    band-pass 0.720, gradient 0.608.
- Interpretation: the filters are complementary across subjects rather than a
  reliable handcrafted bowel segmenter. They are appropriate policy features,
  but not yet a defensible standalone intrinsic reward.

## 2026-07-27 — M5: reward-step profiling and long GRU launch

- Goal: remove environment/reward overhead without changing observations,
  action projection, path geometry, Dice, GDT, or reward values, then start
  the first long reward-supervised/inference-clean GRU run.
- Profile protocol:
  - real warm-cache case `s0120` on the RTX 5070 Ti;
  - reward-supervised clean-policy environment, 24 mm patch, 6 mm maximum
    displacement, and 9 mm path radius;
  - 256 random environment steps after warm-up, synchronized CUDA wall time;
  - the profiler is preserved as `scripts/profile_navigator_steps.py`.
- Baseline environment-only result: `1,179.1` steps/s, or `0.848 ms/step`.
  The largest measured components were path-tube dilation/uniquing
  (`0.234 ms/attempted step`), state patch construction (`0.152 ms/step`),
  reward calculation (`0.129 ms/step`), and action projection
  (`0.094 ms/step`). This excludes policy inference and PPO optimization.
- Accuracy-neutral changes:
  - extract all four static filter patches in one multichannel slice rather
    than four independent slices;
  - bypass line rasterization and all-ones mask indexing for clean-policy
    bounds-only action validation;
  - cache the exact unique dilated relative geometry of each short rasterized
    line and translate/clip it per step;
  - retain Dice coverage as a Python scalar rather than allocating and then
    synchronizing a one-value CUDA tensor.
- Steady-state result after the finite line-geometry cache warmed:
  `1,590.3` steps/s, or `0.629 ms/step`, a `34.9%` environment-step throughput
  increase. Reward calculation itself fell to `0.117 ms/step`; path updates
  fell to `0.212 ms` per accepted movement. The first few hundred diverse
  movements populate the cache, so cold improvement is smaller.
- Equivalence gates:
  - batched multichannel patch extraction is byte-exact against independent
    per-channel extraction, including volume boundaries;
  - cached path construction matches an independent physical-tube
    implementation exactly and preserves path/Dice counters;
  - clean-policy adversarial label-invariance and reward-separation tests
    continue to pass;
  - full suite: 58 tests, 18 warnings, and two subtests passed.
- Seed provenance for this supervised protocol is frozen by
  `scripts/export_navigator_reward_seeds.py`. It exports the first anatomical
  endpoint from each preflight cache in native XYZ order and records explicitly
  that these are label-derived training/validation seeds. They must never be
  described as operator-supplied or annotation-free.
- Planned long run:
  - service `navigator-gru-reward-supervised-3m-v1.service`;
  - from-scratch shared-encoder GRU PPO, hidden width 256, recurrent sequences
    of 64, BF16 AMP;
  - 3,000,000 requested frames, checkpoints approximately every 50k frames,
    three-case validation approximately every 500k frames;
  - immutable v2 330-case training manifest and three-case validation smoke
    manifest;
  - exact Dice-potential and GDT-potential reward supervision, with no GT
    policy inputs or GT action projection.
- Launch:
  - active since `2026-07-27 22:24 EEST` after correcting the configured
    `uv` path to `/usr/bin/uv`;
  - TensorBoard run:
    `/home/matey/project/segmentor/checkpoints/navigator-gru-reward-supervised-3m-v1/nnunet-actual/tensorboard/20260727-222409-906639`;
  - first 6,144 frames completed at roughly `340–360` end-to-end training
    steps/s, including PPO updates and cold filter-cache work;
  - observed total GPU use was about `10.4 GiB / 15.9 GiB`, with no OOM;
  - early completed-episode training Dice values are not validation results
    and ranged around `0.5–0.7`; reward remained negative and endpoint-plus-
    Dice traversal success is the decisive metric.

### M5 final result

- The run was stopped deliberately at `1,254,400` frames because its two
  fixed three-case validations were bit-for-bit identical:
  - frames `500,736` and `1,001,472`;
  - mean Dice `0.0115325672`;
  - traversal and endpoint success `0/3`;
  - mean endpoint distance `185.7095 mm`;
  - all trajectories exhausted the `2,048`-step horizon.
- A separate exact Beta-mode evaluation of checkpoint `1,254,400` was also
  unsuccessful: mean Dice `0.0080688884`, traversal and endpoint success
  `0/3`, and mean endpoint distance `176.1807 mm`.
- Training-only Dice near `0.156` at roughly one million frames was therefore
  not predictive of held-out traversal. Seven training episodes were logged as
  successes among 493 completed-episode records, with none after frame
  `534,528`.
- Conclusion: neither more frames nor switching deterministic evaluation from
  the Beta mean to the paper-style Beta mode fixed the stalled clean-policy
  run.

## 2026-07-27 — M6: exact discrete displacement PPO

- Motivation: the continuous Beta policy emits a real-valued 3-D action, but
  the environment rounds/projects it to one integer voxel displacement. Many
  distinct action densities therefore induce the same transition, producing a
  discontinuous and highly redundant optimization problem. This is not an
  invalid score-function estimator, but it needlessly separates PPO likelihood
  from the behaviorally meaningful move probability.
- Added an opt-in categorical recurrent actor:
  - one category for every nonzero integer displacement in
    `[-max_step_vox, max_step_vox]^3`;
  - `728` categories for the current four-voxel maximum;
  - the sampled category maps directly to its exact displacement;
  - the initial bias assigns equal total probability to each Chebyshev
    step-length shell, preventing the larger number of long-move categories
    from dominating the initial policy;
  - deterministic evaluation uses categorical mode/argmax;
  - the original Beta policy remains the default for compatibility.
- Verification:
  - focused remote suite: 33 tests, 18 warnings, and two subtests passed;
  - full remote suite: 61 tests, 18 warnings, and two subtests passed;
  - 256-frame real-data CUDA smoke completed end to end at `206–270`
    steps/s with `1,098.4 MiB` peak allocated and `2,264 MiB` reserved.
- Controlled real-data ablation `navigator-gru-categorical-gdt1-102k-v1`:
  - same seed, 330-case training manifest, held-out cases `s1389`, `s0224`,
    and `s0120`, GRU-256, 64-step recurrent sequences, BF16, and reward
    protocol as M5;
  - only the action distribution changed from Beta to categorical;
  - `102,400` frames completed in `5m04s`, with `5,051.8 MiB` peak CUDA
    allocated and `6,722 MiB` reserved;
  - at frames `25,600` and `51,200`: mean Dice `0.0148802710`, endpoint
    distance `175.5833 mm`, success `0/3`;
  - at frames `76,800` and `102,400`: mean Dice `0.0115325672`, endpoint
    distance `185.7095 mm`, success `0/3`;
  - top categorical action probability rose from `0.0139` to `0.0509`, while
    deterministic behavior converged to a generic one-voxel direction.
- Interpretation: exact move likelihoods produce a small early improvement but
  do not by themselves solve the credit-assignment problem. The final policy
  collapses to the same poor held-out trajectory as M5.
- Follow-up `navigator-gru-categorical-gdt5-102k-v1` changes only normalized
  geodesic-progress reward scale from `1` to `5`; it was launched after the
  GDT-1 run completed.
- Operational fix: the nnU-Net wrapper now chooses categorical mode by default
  when `NAVIGATOR_ACTION_DISTRIBUTION=categorical`, while preserving Beta mean
  as the default for Beta policies.

### M6 follow-up: reward scale and factorized exact actions

- `navigator-gru-categorical-gdt5-102k-v1` completed the same `102,400`-frame
  protocol in `5m08s`. Increasing only the normalized GDT potential from 1× to
  5× did not help:
  - Dice by checkpoint: `0.014880`, `0.009124`, `0.008069`, `0.008272`;
  - endpoint distance: `175.58`, `176.09`, `176.18`, `175.88 mm`;
  - traversal success remained `0/3`;
  - final top-action probability was `0.0535`.
- The preserved exact first-step audit
  `scripts/audit_navigator_reward_landscape.py` evaluated all 728 moves at the
  three held-out start seeds:
  - `s0120`: 169 positive-reward moves, best reward `0.0709`;
  - `s0224`: 227 positive-reward moves, best reward `0.1834`;
  - `s1389`: 298 positive-reward moves, best reward `2.2656`;
  - the initial shell-balanced expected rewards were nevertheless
    `-0.5613`, `-0.6362`, and `-0.3202`.
- This rules out an absent local reward gradient. It also exposed a categorical
  design problem: equal *total* mass per Chebyshev step-length shell makes each
  individual one-voxel move roughly 15 times more likely than each four-voxel
  move, so categorical mode is structurally biased toward short steps.
- Added `factorized_categorical`, which emits three nine-way categorical
  coordinates for the current `[-4,4]` displacement range:
  - only 27 logits instead of 728;
  - exact integer actions and an exact scalar joint log probability obtained by
    summing the three factor log probabilities;
  - no continuous rounding and no shell-count bias;
  - one zero-displacement combination remains and receives the existing
    explicit zero-movement penalty.
- Verification:
  - focused suite: 35 tests, 18 warnings, and two subtests passed;
  - full suite: 63 tests, 18 warnings, and two subtests passed;
  - real-data 256-frame CUDA collection plus PPO backward/update completed in
    `10.9s`, with `2,062.2 MiB` allocated and `2,948 MiB` reserved.
- Matched run `navigator-gru-factorized-gdt1-102k-v1` was launched with the
  original GDT-1 reward to isolate the action-factorization change.

### M6 critical audit correction: hidden GT-constrained dynamics

- The first factorized run peaked at mean held-out Dice `0.017573` at frame
  `25,600`, then declined to `0.010209` by frame `76,800`; success stayed
  `0/3`. A 10× lower critic coefficient briefly reduced endpoint distance to
  `165.05 mm` but likewise collapsed and never traversed a case.
- Investigation of the repeated deterministic trajectories found a protocol
  violation in `_calculate_reward`: when a clean-policy action landed outside
  the supervised mask, the mask-constrained GDT was infinite. The function
  returned an empty rasterized segment, and `_step` interpreted that as a
  rejected move. Thus labels did not appear in the observation or explicit
  action projector, but they still silently constrained the transition.
- This explains the suspiciously high training Dice and repeated validation
  paths: the agent was often pinned at the seed or forced to remain inside the
  target. **All reward-supervised results before this correction are invalid as
  evidence for an inference-clean policy.**
- Fix:
  - an infinite supervised GDT now adds the configured negative reward;
  - the bounds-valid line segment is still returned and the action executes;
  - the last finite GDT potential is retained so returning to the target cannot
    create a leave/re-enter reward cycle;
  - target Dice and out-of-target penalties remain reward terms only.
- The adversarial separation regression now uses two masks for the same image
  and action where the next GDT is finite in one case and infinite in the
  other. It asserts identical policy observations and the exact same executed
  destination `(8,8,10)`, while requiring different rewards. This specific
  case failed before the fix.
- Actor/critic optimization cleanup:
  - added optional `separate_actor_critic_losses`;
  - when enabled, critic gradients update only the value head, while the shared
    encoder/GRU receives actor gradients;
  - actor/shared and critic-only parameter sets are clipped separately, so a
    large value-head norm cannot rescale the actor gradient;
  - exposed entropy coefficient, value coefficient, and maximum gradient norm
    through the systemd training wrapper.
- Verification after the dynamics fix:
  - focused suite: 35 tests, 18 warnings, and two subtests passed;
  - full suite: 63 tests, 18 warnings, and two subtests passed.
- The invalid separated-loss run was stopped immediately. Replacement
  `navigator-gru-factorized-clean-dynamics-102k-v1` starts from scratch with
  exact factorized actions, truly bounds-only transitions, actor/critic loss
  routing and clipping separated, and the original GDT-1 reward.

## 2026-07-28 — M7: off-target Euclidean recovery reward

- The first honest clean-dynamics diagnostic was stopped after its
  `25,600`-frame validation:
  - mean Dice `0.0193954`;
  - mean endpoint distance `478.317 mm`;
  - traversal and endpoint success `0/3`;
  - validation trajectories used all `2,048` steps and finished with returns
    around `-2,700`.
- Unlike the invalid pre-fix runs, the agent now genuinely wandered hundreds
  of millimetres from the seed. This validated the bounds-only transition fix
  but exposed a reward-support problem: GDT is infinite outside the target, so
  the old flat failure penalty supplied no direction back toward the bowel.
- Added a reward-only Euclidean distance-to-target potential:
  `scale * (distance_t - distance_t+1) / max_3d_step`. The distance transform
  is zero inside the mask and uses the volume's physical voxel spacing. Thus:
  - moving farther from the target is negative;
  - moving toward it is positive;
  - a leave/return excursion contributes zero potential reward;
  - the executed action remains determined solely by the image bounds.
- The maximum-step normalizer is the true diagonal 3-D displacement
  (`10.392 mm` for the current four-voxel, 1.5-mm setup), not the 6-mm
  per-axis/Chebyshev limit.
- Every executed rasterized segment that touches background also pays the
  fixed `r_val1` penalty. Checking the whole segment prevents a long discrete
  move from crossing background and landing on a nearby loop without cost.
  Consequently, a leave/return cycle is strictly negative after fixed costs
  even though the directional potential itself telescopes to zero.
- When Euclidean recovery is enabled, it replaces the non-directional
  `r_val2` penalty for infinite GDT. The legacy flat penalty remains when the
  recovery map is disabled.
- Recovery is computed against the start/goal-connected GDT support
  intersected with the bowel mask, not the union of every labeled component.
  This prevents a disconnected annotation island from becoming a zero-distance
  refuge with infinite goal GDT. Cached maps are versioned as
  `target_distance_connected_edt_v2.nii`. CuCIM computes them on the GPU when
  available; annotation-free mode rejects any nonzero recovery scale because
  this reward requires labels during training.
- Regression and integration gates:
  - same image/action with finite versus infinite supervised GDT executes the
    identical transition;
  - leaving is negative, returning is directionally positive, and their
    combined reward is negative;
  - an interior-to-interior action crossing one background voxel is penalized;
  - full remote suite: 69 tests, 18 warnings, and two subtests passed.
- A 256-frame real-case CUDA smoke completed in `9.737s`, generated three
  prefetched EDT caches, and used `645.6 MiB` peak allocated / `776.0 MiB`
  peak reserved CUDA memory.
- Preliminary ablation
  `navigator-gru-factorized-recovery1-102k-v1` was stopped before its first
  validation after the disconnected-component loophole above was found.
  Replacement `navigator-gru-factorized-recovery1-102k-v2` is launched from
  scratch:
  GRU-256, exact factorized categorical actions, separated actor/critic
  gradient routing, recovery scale 1, GDT scale 1, the immutable 330-case
  training manifest, and only the same three held-out validation cases. Its
  connected-map 256-frame CUDA smoke completed in `9.614s` with `646.2 MiB`
  allocated / `776.0 MiB` reserved. The full run's TensorBoard directory is
  `/home/matey/project/segmentor/checkpoints/navigator-gru-factorized-recovery1-102k-v2/nnunet-actual/tensorboard/20260728-002643-914233`.
- The v2 run reached its first validation trigger but failed during the second
  held-out case with a real CUDA OOM: trainer `4.40 GiB`, training-prefetch
  worker `5.48 GiB`, validation-prefetch worker `3.89 GiB`, followed by a
  requested `1.09 GiB` patch allocation. The workers had already copied their
  results to NumPy but CuPy retained the temporary EDT/filter allocations in
  its caching pools.
- Added an accuracy-neutral worker cleanup after each subject is fully
  materialized: both CuPy device and pinned-memory pools are explicitly
  released. The failed v2 checkpoint is not continued; the corrected run is
  restarted from scratch after a validation-inclusive memory gate.
- Validation-inclusive gate `navigator-recovery-memory-gate-v3`:
  - full suite before launch: 70 tests, 18 warnings, and two subtests passed;
  - completed 25,600 frames plus all three held-out 2,048-step rollouts in
    `1m54s` wall time without OOM;
  - peak trainer CUDA memory was `4,976.1 MiB` allocated / `5,680 MiB`
    reserved, while the completed CuPy workers retained no visible CUDA
    allocation;
  - mean Dice `0.0277485`, mean endpoint distance `451.465 mm`, endpoint and
    traversal success `0/3`;
  - per-case Dice for `s1389`, `s0224`, and `s0120`: `0.010745`,
    `0.056525`, and `0.015975`;
  - per-case endpoint distance: `149.226`, `840.200`, and `364.969 mm`.
- Relative to the matched honest no-recovery diagnostic at 25,600 frames,
  recovery improved mean Dice from `0.0193954` to `0.0277485` and endpoint
  distance from `478.317` to `451.465 mm`. This is encouraging but remains far
  below the 0.40 Dice and complete-traversal research target.
- Full from-scratch run
  `navigator-gru-factorized-recovery1-102k-v3` was launched only after this
  gate passed, with validations/checkpoints every 25,600 frames and no more
  than the fixed three validation cases.
- V3 validation at frame `25,600`:
  - mean Dice `0.0448746`, mean endpoint distance `281.253 mm`, endpoint and
    traversal success `0/3`;
  - per-case Dice (`s1389`, `s0224`, `s0120`): `0.081250`, `0.031342`,
    `0.022032`;
  - per-case endpoint distance: `145.763`, `336.823`, `361.173 mm`;
  - peak validation-time trainer allocation observed by `nvidia-smi` was about
    `6.1 GiB`, well inside the 15.5-GiB device.
- V3 and the memory gate both use seed 42 but are not bitwise deterministic
  under the CUDA/collector pipeline. Both independently beat the honest
  no-recovery result at the same frame count; v3 improved mean Dice by 2.31×
  and reduced endpoint distance by 197.1 mm. It remains far from success, so
  the run continues to the preregistered 102,400 frames.
- V3 validation at frame `51,200`:
  - mean Dice declined to `0.0217959`, while mean endpoint distance improved
    to `262.729 mm`; endpoint and traversal success stayed `0/3`;
  - per-case Dice: `0.050079`, `0.010589`, `0.004719`;
  - per-case endpoint distance: `243.578`, `272.261`, `272.348 mm`.
- Interpretation at this checkpoint: Euclidean recovery is reducing
  catastrophic drift, but coverage learning is not yet stable. The
  preregistered rank therefore keeps frame 25,600 as best. No hyperparameter is
  changed mid-run.
- V3 validation at frame `76,800`:
  - new best mean Dice `0.0579610`, mean endpoint distance `319.316 mm`, and
    endpoint/traversal success `0/3`;
  - per-case Dice: `0.153130`, `0.011726`, `0.009027`;
  - per-case endpoint distance: `187.668`, `484.760`, `285.521 mm`.
- The mean gain is dominated by `s1389`; the other two held-out cases remain
  near seed-level Dice. Recovery permits real coverage learning on one case
  but has not produced stable cross-subject generalization.
- V3 final validation at frame `102,400`:
  - new best mean Dice `0.0628456`, mean endpoint distance `291.938 mm`, and
    endpoint/traversal success `0/3`;
  - per-case Dice: `0.146759`, `0.015184`, `0.026594`;
  - per-case endpoint distance: `105.534`, `484.760`, `285.521 mm`;
  - the complete run took `7m04.7s` wall time and peaked at `5,455.4 MiB`
    allocated / `6,236 MiB` reserved CUDA memory.
- At 2,048 frames per subject, this pilot sampled only about 50 of the 330
  training cases. It therefore cannot answer the earlier 1–3-million-frame
  question or establish cohort-wide generalization.
- Follow-up `navigator-gru-factorized-recovery1-warm1m-v1` warm-starts the
  policy and value weights from v3's final/best 102,400-frame checkpoint and
  collects one million additional frames:
  - optimizer/scheduler state is intentionally reset because the pilot's
    cosine schedule had already reached its floor;
  - learning rate is reduced from `5e-5` to `2e-5`;
  - all architecture, action, reward, split, and seed settings remain fixed;
  - validation is reduced to four checkpoints (approximately every 256k new
    frames), always on only the same three held-out cases;
  - regular model checkpoints remain approximately every 51k frames.
- The warm run started successfully with the requested configuration and
  loaded the v3 policy/value checkpoint. TensorBoard:
  `/home/matey/project/segmentor/checkpoints/navigator-gru-factorized-recovery1-warm1m-v1/nnunet-actual/tensorboard/20260728-004201-915859`.

### M7 warm-1M interim report at 403,456 frames

- Service `navigator-gru-factorized-recovery1-warm1m-v1` remained healthy at
  `403,456 / 1,000,000` new frames (`40.3%`) with no CUDA or worker failure.
  Observed resident CUDA memory was approximately `6,554 MiB` for the trainer
  and `224 MiB` for the preprocessing worker.
- The configured evaluation cadence produced validations every 128,000 frames,
  rather than the anticipated 256,000, because the trainer's evaluation
  counter advances per PPO update. Each validation still uses only the fixed
  three cases.
- Held-out results:
  - frame `128,000`: mean Dice `0.039465`, endpoint distance `208.487 mm`,
    traversal/endpoint success `0/3`;
  - frame `256,000`: mean Dice `0.060485`, endpoint distance `115.641 mm`,
    traversal/endpoint success `0/3`;
  - frame `384,000`: mean Dice `0.064236`, endpoint distance `122.653 mm`,
    traversal/endpoint success `0/3`.
- At frame 384,000, per-case Dice (`s1389`, `s0224`, `s0120`) was `0.141279`,
  `0.001673`, and `0.049757`; endpoint distance was `54.374`, `105.609`, and
  `207.976 mm`.
- Interpretation: endpoint localization has improved strongly relative to the
  102,400-frame pilot (`291.938` to `122.653 mm`), but mean Dice is essentially
  plateaued around `0.06`, no endpoint is within tolerance, and no traversal
  succeeds. The mean remains dominated by `s1389`; this is not yet evidence of
  cohort-wide tube following or progress toward the 0.40 Dice target.

### M8: bounded episodic spatial exploration

- Motivation: the warm-1M control improves endpoint distance but has not
  generalized its coverage beyond one of the three held-out cases. Exploration
  is therefore a plausible bottleneck, but CT-feature prediction error or
  random-feature novelty could reward scanner texture and unrelated anatomy.
- Design: add an opt-in, label-free, first-visit bonus over 6-mm spatial cells.
  If the agent enters its `n`th new cell in an episode, the added reward is
  `0.05 / sqrt(n)`; revisits receive zero. The reset position is marked visited
  without reward. The construction borrows the episodic novelty principle from
  [E3B](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f4f79698d48bdc1a6dec20583724182b-Abstract-Conference.html),
  while using exact controllable position instead of a learned embedding.
  [RE3](https://proceedings.mlr.press/v139/seo21a.html) and
  [Revisiting Intrinsic Reward](https://openreview.net/forum?id=j3GK3_xZydY)
  support episodic state novelty as a useful, lightweight exploration signal,
  but no claim is made that this exact cell counter reproduces those
  algorithms.
- Reward-hacking controls:
  - no segmentation, GDT, endpoint, subject ID, or validation state enters the
    bonus;
  - the bonus is first-visit-only and decreases within every episode;
  - at the preregistered scale, a newly visited off-target cell receives at
    most `+0.05`, while the existing off-target penalty alone is `-0.25`
    before step and wall costs, so novel background remains immediately
    unprofitable;
  - the intrinsic component is logged separately as
    `train/episodic_cell_reward`.
- Implementation adds validated CLI/configuration fields, forwards them
  through the nnU-Net and systemd launchers, emits the component in transition
  info, and leaves the feature disabled by default for protocol
  reproducibility.
- Verification:
  - a focused environment suite passed `44` tests;
  - the complete suite passed `73` tests and two subtests with `18`
    deprecation warnings;
  - dedicated tests confirm first-visit behavior, inverse-square-root decay,
    zero revisit reward, invalid configuration rejection, and negative total
    reward for a novel off-target cell.
- Launched overnight ablation
  `navigator-gru-episodic-cell-warm8m-v1`:
  - wait for the matched warm-1M control to finish and require its
    `final_model_torchrl.pth` before starting;
  - warm-start only policy/value weights, resetting optimizer state;
  - collect 8,000,000 additional frames with learning rate `1e-5`;
  - preserve the GRU, factorized categorical action distribution, separated
    actor/critic loss updates, reward-supervised inputs, fixed manifests, and
    fixed three-case validation cohort;
  - use an evaluation interval of 2,000 epoch updates (approximately every
    512k frames under the observed counter semantics) and save every 500
    updates (approximately every 128k frames).
- The systemd queue handed off cleanly after the warm-1M control stopped:
  - the required final control checkpoint existed before launch;
  - the intrinsic run was active at 384,000 frames with one trainer using
    approximately 5,786 MiB of CUDA memory and no competing training process;
  - TensorBoard:
    `/home/matey/project/segmentor/checkpoints/navigator-gru-episodic-cell-warm8m-v1/nnunet-actual/tensorboard/20260728-014524-917157`.

### M8 outcome and deployment failure

- The episodic-cell run remained numerically unstable:
  - its best validation at 3,072,000 frames reached mean Dice `0.122161`,
    dominated by `s1389=0.347935`; `s0224=0.014689` and
    `s0120=0.003858` did not generalize;
  - its best mean endpoint distance was `110.456 mm` at 4,096,000 frames;
  - traversal and endpoint success remained `0/3`;
  - the final completed validation at 4,608,000 frames had regressed to mean
    Dice `0.010505` and endpoint distance `262.602 mm`.
- The service stopped at checkpoint 5,504,000 rather than 8,000,000. While
  developing M9, source files were synchronized into the same checkout used by
  the live process. A validation DataLoader worker imported the newer dataset
  module but received the older in-memory `Config`, which lacked the new
  derived `needs_target_distance` field. This hot-code deployment mistake
  caused validation to fail and the service to exit. It was not an OOM or an
  algorithmic terminal condition.
- The 5,504,000 checkpoint and 3,072,000 best checkpoint are preserved.
  Future source synchronization must use an idle or separate checkout. Dataset
  loading now also has a backward-compatible derived-field fallback.

### M9: numerically audited supervised reward contract

- Inspection of the exact cached feature volumes showed that the Meijering
  response was not consistently aligned with bowel:
  - dark-tube mean response was lower inside than outside for `s1389` and
    `s0120`, but higher inside for `s0224`;
  - the navigation filters were already robustly normalized to `[0,1]`, after
    which the reward transformed `[0,0.1]` to `[0,1]` a second time;
  - this saturated `23.7%`, `47.7%`, and `48.3%` of the complete validation
    volumes and produced a measured mean wall penalty of approximately `-155`
    per episode. M9 therefore sets the image-filter reward scale to zero while
    retaining the channels as policy inputs.
- The historical GDT normalization made total monotonic endpoint progress
  worth only `+1` per subject. Across cached finite cases, start-to-end GDT
  ranged from `0` to `454.51 mm` (median `188.74 mm`), so identical physical
  actions received subject-dependent hidden scales.
- M9 reward protocol:
  - GDT progress is normalized by the maximum physical action length
    `sqrt(3)*6 = 10.392 mm`, with scale `0.1`;
  - Euclidean recovery-potential scale is reduced from `1.0` to `0.05`;
  - the binary off-target penalty is removed;
  - a persistent segment-distance penalty is
    `-0.1 * min(max_distance_along_segment / 30 mm, 1)`;
  - any segment crossing background is forbidden from receiving positive GDT,
    Dice, or supervised episodic-cell reward, while negative Dice damage is
    retained;
  - episodic first-cell scale is reduced from `0.05` to `0.01`;
  - terminal success is a fixed `+50` only when endpoint and Dice threshold
    both pass; terminal failure is `0`, avoiding the previous `-52` critic
    shock;
  - fixed valid-step cost remains `-0.01`.
- Canonical numerical audit:
  - 6-mm in-target forward/new-cell transition: `+0.057735`;
  - in-target forward/backward cycle: `-0.010000`;
  - 1.5-mm leave/return cycle: `-0.030000`;
  - 6-mm leave/return cycle: `-0.060000`;
  - 1.5-mm cross-loop background gap: `-0.015000`;
  - tangential motion 6 mm from target: `-0.030000` per transition;
  - tangential motion at or beyond 30 mm: `-0.110000` per transition;
  - maximum 2,048-step episodic-cell return: `+0.890604`, versus fixed step
    cost `-20.48`.
- A reusable `scripts/audit_navigator_reward_contract.py` fails if valid
  forward progress is not positive, a cross-loop shortcut is profitable, or a
  canonical inside/outside cycle has non-negative return.
- All reward terms are emitted and logged separately for empirical accounting:
  invalid, GDT, recovery, persistent target distance, step, image wall,
  off-target, coverage, image novelty, curvature, episodic cell, and terminal.
- Verification: the numerical audit passed, focused reward/environment tests
  passed, and the full suite passed `80` tests plus two subtests with `18`
  deprecation warnings.

### M9.1 real-data 25.6k reward gate

- `navigator-gru-reward-contract-25k-v1` completed from scratch in `2m13s`
  without OOM. Peak CUDA memory was `4,995.1 MiB` allocated / `5,690 MiB`
  reserved.
- Three-case validation at frame 25,600:
  - mean Dice `0.018376`;
  - mean endpoint distance `336.200 mm`;
  - endpoint and traversal success `0/3`.
- Separate component logs rejected the nominal 30-mm distance protocol before
  a long run:
  - mean total reward per step: `-0.103460`;
  - persistent target-distance term: `-0.093290`;
  - step cost: `-0.010000`;
  - GDT: `+0.000008`;
  - recovery: `-0.000350`;
  - Dice coverage: `+0.000142`;
  - episodic cell: `+0.000030`;
  - wall and binary off-target terms: exactly zero.
- Thus the 30-mm radius saturated almost immediately after a random policy
  left the bowel and recreated a nearly constant penalty. No longer run is
  justified with this setting.
- M9.2 expands the physical distance range to 600 mm. Across the audited cache,
  the maximum target distance was approximately 643 mm, so this retains a
  graded signal through nearly the complete volume while bounding the maximum
  penalty near `-0.1`. Numerically:
  - 6 mm from target: distance cost `-0.001`;
  - 30 mm: `-0.005`;
  - 150 mm: `-0.025`;
  - 300 mm: `-0.050`;
  - at or beyond 600 mm: `-0.100`.

### M9.2 real-data 25.6k reward gate

- `navigator-gru-reward-contract-25k-v2` completed from scratch in `2m19s`
  without OOM. Peak CUDA memory was `5,006.1 MiB` allocated / `5,730 MiB`
  reserved.
- The component distribution was materially less saturated:
  - mean total reward per step: `-0.030932`;
  - persistent target-distance term: `-0.020862`;
  - fixed step cost: `-0.010000`;
  - GDT: `+0.000021`;
  - recovery: `-0.000476`;
  - Dice coverage: `+0.000337`;
  - episodic cell: `+0.000048`;
  - wall, binary off-target, and terminal terms: exactly zero during training;
  - mean value loss was `0.135334`, versus `0.750226` in M9.1.
- Three-case validation at frame 25,600:
  - mean Dice `0.047421`, versus `0.018376` in M9.1;
  - mean endpoint distance `309.472 mm`, versus `336.200 mm`;
  - endpoint and traversal success remained `0/3`;
  - per-case Dice (`s1389`, `s0224`, `s0120`): `0.120505`,
    `0.007085`, `0.014674`;
  - per-case endpoint distance: `185.794`, `398.351`, `344.272 mm`.
- Interpretation: no individual reward term is saturated at the old
  `-0.1` level, critic scale is substantially reduced, and the short result is
  competitive with earlier 25.6k pilots. Generalization remains absent and
  `s1389` still dominates, so this supports only a 102.4k from-scratch
  comparison, not a long run.

### M10: BOMOPI-only movement and reward gate

- Scope is restricted to `data/bomopi_resampled2`: 22 locally resampled
  subject directories containing CT plus small-bowel, duodenum, and colon
  labels. The nnU-Net cohort is not opened by this protocol.
- The source audit found three byte-identical CT/label pairs:
  `pt5=pt8`, `pt6=pt10`, and `pt11=pt16`. The fixed seed places every member
  of these pairs in training in the unfiltered 22-directory split.
- Full preflight rejected `pt13` and `pt22` because their label-derived
  anatomical endpoints are disconnected in the small-bowel mask. End-to-end
  traversal is impossible for those targets. Removing only those two cases
  would put `pt16` in validation while byte-identical `pt11` remained in
  training.
- The actual gate therefore uses the immutable manifest
  `experiments/navigator-bomopi-v1/eligible.txt`: 17 unique, traversable
  anatomies exposed through a non-destructive symlink view. It excludes
  impossible `pt13`/`pt22` and duplicate copies `pt8`/`pt10`/`pt16`; the
  transferred source directory remains untouched.
- Source volumes are synchronized to the CUDA checkout without the old cache
  directories. Versioned GDT, target-distance, wall, and navigation-filter
  caches are recomputed by the current code.
- The persistent systemd gate runs `scripts/preflight_navigator_bomopi.py`
  across every eligible directory before starting PPO; any invalid spacing, shape,
  navigation-filter tensor, endpoint, or disconnected endpoint path aborts the
  service rather than failing partway through training.
- Before launch, exact zero movement was corrected: `(0, 0, 0)` now remains
  stationary and receives the configured invalid-action penalty instead of
  being silently executed as a maximum `+X` step.
- The legacy subject loader now computes the same four image-only navigation
  filter channels used by clean nnU-Net reward-supervised training. Labels
  remain available to the reward/evaluator, not to policy observations or
  movement projection.
- The preregistered first gate is `navigator-bomopi-gru-102k-v1`:
  - 90/10 seeded split, yielding 15 training subjects and 2 validation
    subjects (`pt14`, `pt18`) with no duplicate-patient leakage;
  - GRU-256 and exact factorized categorical voxel actions;
  - M9.2 reward contract: max-step-normalized GDT `0.1`, recovery potential
    `0.05`, 600-mm graded target-distance penalty `0.1`, coverage `50`,
    episodic first-cell bonus `0.01`, step cost `0.01`, fixed success `+50`,
    zero wall/binary-off-target/curvature terms;
  - entropy coefficient reduced from `0.003` to `0.0005` because prior runs
    remained almost maximally entropic and the entropy term dominated much of
    the actor loss;
  - 102,400 frames from scratch, one final two-case validation, TensorBoard,
    and saved validation paths.
- The v1 service intentionally stopped in preflight before PPO:
  `pt13` and `pt22` had disconnected anatomical endpoints. This exposed the
  duplicate-patient validation leak that a naive two-case deletion would have
  created.
- Replacement service `navigator-bomopi-gru-102k-v2` uses the 17-case
  immutable manifest. Preflight passed every eligible case and PPO started
  from scratch on the exact 15/2 split above. Initial CUDA allocation was
  approximately `2.3 GiB`, with no OOM or loader failure.
- TensorBoard:
  `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-102k-v2/data/bomopi_resampled2_unique-v1/tensorboard/20260728-144841-925634`.
- Saved validation paths:
  `/home/matey/project/segmentor/results/navigator_bomopi/gru-102k-v2-validation`.
- V2 completed 102,400 frames in `5m18s` wall time with peak CUDA memory
  `1,942.3 MiB` allocated / `2,608 MiB` reserved. Final two-case validation:
  - mean Dice `0.020817`, endpoint distance `290.766 mm`, and traversal /
    endpoint success `0/2`;
  - `pt14`: Dice `0.009387`, endpoint distance `237.403 mm`;
  - `pt18`: Dice `0.032247`, endpoint distance `344.128 mm`.
- V2 is rejected as a movement diagnostic. The factorized mode became
  positively biased (recent action modes approximately `+3,+4,+4`) and drove
  both cases to the positive volume boundary. `pt14` spent `99.90%` and
  `pt18` `99.02%` of their saved paths on a boundary; both had only two unique
  positions and `100%` immediate reversals over their final 512 positions.
  The clean-policy projector was reflecting an impossible outward ray into an
  opposite one-voxel fallback, after which the policy moved outward again.
- Clean-policy projection now shortens a requested ray only while preserving
  its direction. If no forward displacement is in bounds, it executes zero
  movement and applies the existing invalid-action penalty; the opposite
  tangent fallback remains only for legacy mask-constrained dynamics.
- Matched replacement `navigator-bomopi-gru-102k-v3` changes only this boundary
  behavior. Its reward, optimizer, seed, split, architecture, and 102,400-frame
  budget remain identical to v2.
- V3 completed in `5m05s` wall time with peak CUDA memory `1,942.6 MiB`
  allocated / `2,610 MiB` reserved. Final two-case validation:
  - mean Dice `0.007603`, endpoint distance `158.943 mm`, and traversal /
    endpoint success `0/2`;
  - `pt14`: Dice `0.003503`, endpoint distance `133.002 mm`;
  - `pt18`: Dice `0.011704`, endpoint distance `184.884 mm`.
- The movement correction passed its empirical acceptance test:
  - neither v3 path touched a volume boundary, versus approximately 99% of
    both v2 paths;
  - final-512 immediate reversal rate was `0%` for both subjects, versus
    `100%` in v2;
  - `pt14` visited 64 unique positions (36 in its final 512 recorded
    positions), and `pt18` visited 51 (19 in its final 512). Thus the exact
    boundary loop is gone, although substantial local revisitation remains.
- Training diagnostics also became healthier:
  - invalid-action incidence was `1.737%` over the complete run, `0.762%`
    over the final ten updates, and `0.391%` in the final batch, showing that
    PPO learned to avoid some newly explicit boundary failures;
  - the graded target-distance cost improved from a complete-run mean
    `-0.009921` to `-0.005734` over the final ten updates and `-0.002490` in
    the final batch;
  - value loss fell to `0.038760` in the final batch, whereas v2 ended at
    `0.563134`;
  - GDT progress remained slightly negative and validation Dice regressed, so
    this is evidence for corrected dynamics and improved localization, not
    successful bowel following.
- V3 artifacts:
  - final model:
    `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-102k-v3/data/bomopi_resampled2_unique-v1/final_model_torchrl.pth`;
  - TensorBoard:
    `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-102k-v3/data/bomopi_resampled2_unique-v1/tensorboard/20260728-145851-926661`;
  - validation paths:
    `/home/matey/project/segmentor/results/navigator_bomopi/gru-102k-v3-validation`.

### M10.1: 32-cubed visual context

- The v2/v3 observation patch was only `24 mm = 16 voxels` per side at the
  fixed 1.5-mm isotropic spacing. The next ablation uses
  `48 mm = 32 voxels` per side, exactly `32^3`.
- Patch volume and early convolutional activation volume increase by `8x`.
  The visual encoder's adaptive pooling keeps its parameter count unchanged,
  but a batch-128 run is not assumed safe from the earlier 16-GB OOM.
- The launcher therefore defaults to PPO minibatch 64 for the first CUDA
  memory smoke. No reward, movement, split, seed, GRU, rollout, or optimizer
  setting changes. If peak memory is comfortably below capacity, a larger
  minibatch can be profiled separately rather than changed mid-run.
- CUDA smoke `navigator-bomopi-gru-patch32-smoke4k-v1` completed 4,096 frames
  without an OOM in 26.4 seconds wall time:
  - peak CUDA memory: 10,348.7 MiB allocated, 11,936.0 MiB reserved;
  - model size remained 607,516 trainable parameters;
  - batch 64 therefore leaves enough headroom on the 15.5-GiB GPU for a
    same-shape long run.
- Promoted unchanged to the 102,400-frame gate
  `navigator-bomopi-gru-patch32-102k-v1`.

### M10.2: 32-cubed outcome and GDT-scale ablation

- `navigator-bomopi-gru-patch32-102k-v1` completed 102,400 frames in
  6 minutes 35 seconds:
  - training peak CUDA memory was 10,357.7 MiB allocated and 12,148.0 MiB
    reserved; validation raised the process peak to 11,442.3 MiB allocated and
    12,942.0 MiB reserved;
  - mean validation Dice was `0.035143` over pt14/pt18 (`0.010409`,
    `0.059877`), with zero traversals and 209.383-mm mean endpoint distance;
  - trajectories had 259/361 unique positions and no zero moves or immediate
    reversals. The larger patch improved v3's `0.007603` Dice and severe local
    revisitation, but produced broad unguided exploration rather than traversal.
- The last ten training batches numerically decomposed to:
  - target-distance state cost `-0.014269`;
  - invalid-action cost `-0.014258`;
  - fixed step cost `-0.009857`;
  - coverage shaping `+0.000995`;
  - GDT shaping `-0.000054`.
  The useful directional potential is too small relative to the persistent
  costs once the policy leaves the target.
- Next ablation changes only GDT scale from `0.1` to `1.0`; the launcher exposes
  this as `NAVIGATOR_GDT_REWARD_SCALE`. The exact reward-contract audit gives:
  - 6-mm on-target forward/new: `+0.057735 -> +0.577350`;
  - 6-mm on-target backward/revisit: `-0.067735 -> -0.587350`;
  - forward/backward cycle: still `-0.010000`;
  - 1.5-mm off-target leave/return cycle: still `-0.020500`;
  - 6-mm off-target leave/return cycle: still `-0.022000`;
  - 1.5-mm cross-loop gap: still `-0.010250`.
  Thus the ablation strengthens desired on-target direction without making the
  audited cycles, shortcuts, or off-target wandering profitable.
- `navigator-bomopi-gru-patch32-gdt1-102k-v1` showed that GDT `1.0` alone is
  too endpoint-directed:
  - validation Dice fell to `0.003600` (`0.004020`, `0.003179`);
  - endpoint distance improved from 209.383 mm to 136.005 mm, but neither case
    reached the endpoint;
  - the paths contained only 155/164 valid positions. pt14 ended at `y=130` in
    a size-131 dimension and pt18 at `x=0`, after which deterministic actions
    repeatedly attempted to cross the image boundary;
  - mean validation reward was `-0.924491` per step and total return was
    `-1893.36`, confirming invalid-action collapse rather than traversal.
- The next causal ablation retains GDT `1.0` and changes only the
  target-distance normalization radius from 600 mm to 60 mm, exposed as
  `NAVIGATOR_TARGET_DISTANCE_PENALTY_RADIUS_MM`. At scale `0.1`, a 600-mm radius barely
  distinguishes anatomically serious departures: 6/30 mm cost only
  `-0.001/-0.005`. The 60-mm candidate gives:
  - 6-mm off-target tangent: `-0.020000` including the fixed step cost;
  - 30-mm off-target tangent: `-0.060000`;
  - 1.5-mm leave/return cycle: `-0.025000`;
  - 6-mm leave/return cycle: `-0.040000`;
  - 1.5-mm cross-loop gap: `-0.012500`.
  Recovery remains action-preferential and the audited cycles remain strictly
  negative, while distance from the bowel becomes relevant before a boundary.
- The first attempted r60 systemd run was invalid and is excluded: its saved
  config still recorded 600 mm and its outputs were bit-for-bit identical to
  the GDT-only run. The launcher had introduced
  `NAVIGATOR_TARGET_DISTANCE_RADIUS_MM`, while the systemd environment
  allowlist already used the canonical
  `NAVIGATOR_TARGET_DISTANCE_PENALTY_RADIUS_MM`. The launcher now consumes the
  canonical name, and every promoted run must verify the saved config.
- The corrected run `navigator-bomopi-gru-patch32-gdt1-r60-102k-v2` logged
  `patch_mm=48`, `gdt_scale=1.0`, and `target_distance_radius_mm=60` before
  training. It improved both registered validation objectives:
  - mean Dice `0.079295` and endpoint distance 93.100 mm, versus the 32-cubed
    baseline's `0.035143` and 209.383 mm;
  - pt14: Dice `0.150489`, endpoint distance 38.095 mm;
  - pt18: Dice `0.008101`, endpoint distance 148.106 mm;
  - zero endpoint reaches and zero full traversals.
- Both validation rollouts executed all 2,048 actions without zero movements.
  pt14/pt18 visited 348/385 unique positions, although their last 512 steps
  contracted to 53/17 positions. The cases failed differently:
  - pt14 center positions were on the endpoint-connected target for 97.12% of
    the rollout (99.80% in its last 512) and ended 42.04 geodesic mm from goal;
  - pt18 centers were on-target for only 0.05%, never returned in its last 512,
    and ended 46.79 Euclidean mm from the target mask.
- The four cached image-derived channels are not blank. Around the starting
  32-cubed patch, pt18 nevertheless has roughly half pt14's response amplitude
  across dark/bright tubularity, band-pass, and gradient channels, consistent
  with a harder image-generalization case.
- Training had not plateaued at the gate: completed-episode Dice averaged
  `0.154716`, ended at `0.330172`, and peaked at `0.360105`. Promote the exact
  verified configuration to 512,000 frames, retaining validation/checkpointing
  every 102,400 frames. Do not change recovery scale until this learning curve
  establishes whether pt18 can recover with more policy updates.

### M10.3: 512k learning curve

- Active unit: `navigator-bomopi-gru-patch32-gdt1-r60-512k-v1`.
- Effective configuration was logged before training:
  `steps=512000 patch_mm=48 batch=64 gdt_scale=1.0
  target_distance_radius_mm=60`.
- The 102,400-frame gate is deliberately not comparable to the completed
  102,400-frame schedule at the same frame count: cosine annealing spans the
  full 512,000 frames, so learning rate remains near its initial value rather
  than reaching the 5e-6 minimum.
- First held-out gate:
  - mean Dice `0.003759` (pt14 `0.003726`, pt18 `0.003791`);
  - mean endpoint distance 192.245 mm (155.176, 229.314);
  - zero endpoint reaches and traversals.
- This gate is worse than the short schedule, but the preregistered long curve
  continues through later annealing checkpoints. The trainer retains the
  validation-ranked best checkpoint, so continuing does not discard an earlier
  better long-run state.

### M10.4: 512k completion and collapse diagnosis

- The 512k service completed in 33 minutes 21 seconds without an OOM. Held-out
  gates were:
  - 102.4k: Dice `0.003759`, endpoint distance 192.245 mm;
  - 204.8k: Dice `0.027686`, endpoint distance 232.481 mm;
  - 307.2k: Dice `0.012548`, endpoint distance 200.090 mm;
  - 409.6k: Dice `0.032446`, endpoint distance 203.701 mm;
  - 512.0k: Dice `0.018795`, endpoint distance 206.663 mm.
  No endpoint reach or traversal occurred. The long-run best
  `checkpoint_409600best.pth` remains worse than the short-schedule champion
  (`0.079295`, 93.100 mm).
- This was deterministic policy collapse, not profitable reward hacking:
  - joint maximum action probability rose from `0.0087` at 102.4k to `0.4412`
    at 512k, while logit standard deviation rose from `0.455` to `2.200`;
  - entropy loss magnitude fell from `0.003150` to `0.000978`;
  - value loss rose from `0.130` to `3.017`;
  - final GDT and coverage shaping were effectively zero while target-distance
    cost saturated near `-0.093`;
  - both final validation paths had exactly two unique positions in their last
    512 steps and 100% immediate reversals. On-target center occupancy was only
    0.39%/0.20%.
- The PPO KL was not the primary failure: mean `0.00835`, 95th percentile
  `0.01864`, maximum `0.03066`. Entropy starvation and off-target state
  distribution collapse are the stronger observed mechanisms.

### M11: Overnight anti-collapse matrix

- Preserve the short-schedule champion and add opt-in controls:
  - `NAVIGATOR_TARGET_RECOVERY_REWARD_SCALE`;
  - `NAVIGATOR_ENT_COEF`;
  - `NAVIGATOR_LR_ANNEAL_TIMESTEPS`, which freezes cosine annealing at its
    minimum instead of silently stretching a validated short schedule;
  - `NAVIGATOR_TARGET_KL`, which can skip remaining PPO epochs after excessive
    rollout KL.
- Reward audit for recovery scale `0.2`, GDT `1.0`, radius 60 mm:
  - 1.5-mm leave `-0.041368`, return `+0.016368`, cycle `-0.025000`;
  - 6-mm leave `-0.135470`, return `+0.095470`, cycle `-0.040000`;
  - cross-loop gap remains `-0.012500`;
  - inside forward/backward cycle remains `-0.010000`.
  Recovery becomes locally clear without making an excursion profitable.
- Queue three seed/split-matched 102.4k causal gates:
  1. recovery `0.2`, entropy `0.0005`;
  2. recovery `0.05`, entropy `0.003`;
  3. recovery `0.2`, entropy `0.003`.
  All retain 32-cubed inputs, GDT `1.0`, radius 60 mm, and the short 102.4k
  annealing horizon.
- Conditional supervised-input fallback authorized by the user: for the first
  four hours, keep policy observations image-only. If the best held-out result
  remains below 20% Dice with zero traversal, a separately named GT-mask-input
  baseline may be started. It must never be reported as annotation-free or
  replace the image-only champion.
- Completed 102.4k matrix, all seed/split/architecture matched:
  - recovery `0.2`, entropy `0.0005`: Dice `0.044395`
    (`0.075525`, `0.013265`), endpoint 106.542 mm;
  - recovery `0.05`, entropy `0.003`: Dice `0.015460`
    (`0.004307`, `0.026614`), endpoint 213.497 mm;
  - recovery `0.2`, entropy `0.003`: Dice `0.099903`
    (`0.122358`, `0.077448`), endpoint 114.577 mm;
  - recovery `0.1`, entropy `0.0005`: Dice `0.011585`
    (`0.003954`, `0.019216`), endpoint 148.411 mm;
  - recovery `0.05`, entropy `0.001`: Dice `0.034132`
    (`0.042663`, `0.025601`), endpoint 204.157 mm;
  - recovery `0.1`, entropy `0.001`: Dice `0.069973`
    (`0.080038`, `0.059908`), endpoint 152.104 mm.
- Recovery and entropy interact nonlinearly. Strong settings alone trade one
  validation case against the other, whereas recovery `0.2` plus entropy
  `0.003` is the first balanced image-only policy and becomes the new champion
  at `0.099903` Dice. It still has zero endpoint reaches/traversals, and its
  endpoint distance is worse than the former champion's 93.100 mm.

### M12: Selected million-frame image-only run

- Promoted the balanced champion to
  `navigator-bomopi-gru-p32-g1-r60-rec02-e003-1m-v1`:
  - 1,024,000 frames;
  - exact 32-cubed observations;
  - GDT `1.0`, target-distance radius 60 mm, recovery `0.2`;
  - entropy `0.003`;
  - cosine annealing reaches 5e-6 at 102,400 frames and is then frozen;
  - two-case held-out validation every 51,200 frames;
  - target-KL guard remains disabled for this causal run because KL was
    secondary and variable epoch counts would otherwise alter validation
    cadence.
- Effective settings and TensorBoard path were verified from the systemd log
  before training. Validation-ranked best checkpoints are retained throughout.
- The run completed 1,024,000 frames in 1 hour 3 minutes 53 seconds. The
  learning-rate freeze and entropy prevented the earlier collapse:
  - final joint maximum action probability `0.0429` instead of `0.4412`;
  - recent completed-training Dice `0.2062`;
  - recent GDT shaping `+0.06084`;
  - final value loss `0.694` instead of `3.017`.
- Best held-out gate was 716,800 frames:
  - mean Dice `0.170687` (pt14 `0.244338`, pt18 `0.097035`);
  - endpoint distance 159.411 mm;
  - zero endpoint reaches/traversals.
  The retained model is `checkpoint_716800best.pth`. Final 1,024,000-frame Dice
  regressed to `0.107433`, so the final model is not the champion.
- Resume the exact best checkpoint with optimizer and scheduler state intact to
  a total counter of 1,536,000 frames. It starts from collected frame 716,800
  and update 2,800, adding 819,200 frames at the frozen 5e-6 learning rate.

### M13: Authorized GT-mask-input baseline

- At the four-hour gate, the image-only champion remained below the user's
  conditional 20% threshold and had zero traversal. A supervised-input
  baseline was therefore authorized and implemented as an explicit opt-in
  `observe_segmentation` channel.
- Contract:
  - reward-supervised clean dynamics remain bounds-only;
  - current CT, four image-derived filters, cumulative path, and one local GT
    segmentation patch form seven observation channels;
  - annotation-free mode rejects this flag;
  - all output names include `gtmask`, and results must never be reported as
    image-only or annotation-free.
- CPU reward/environment suite passes 53/53 with an exact channel-content test.
- Queue after the image-only continuation:
  1. 102,400-frame supervised-input gate;
  2. if its final checkpoint exists, resume it to 512,000 total frames with
     validation every 51,200 frames.

### M14: Image-only continuation and stopping result

- Resumed the image-only champion at 716,800 collected frames, preserving its
  optimizer, frozen 5e-6 learning rate, GRU state handling, reward, split, and
  32-cubed observations. The continuation ran to a total counter of 1,536,000
  frames.
- Held-out Dice at the continuation gates was:
  - 768.0k: `0.162786`;
  - 819.2k: `0.134305`;
  - 870.4k: `0.027259`;
  - 921.6k: `0.118093`;
  - 972.8k: `0.102330`;
  - 1,024.0k: `0.100316`;
  - 1,075.2k: `0.038958`;
  - 1,126.4k: `0.046638`;
  - 1,177.6k: `0.089087`;
  - 1,228.8k: `0.080503`;
  - 1,280.0k: `0.079509`;
  - 1,331.2k: `0.059964`;
  - 1,382.4k: `0.066989`;
  - 1,433.6k: `0.065536`;
  - 1,484.8k: `0.086043`;
  - 1,536.0k: `0.054322`.
- No gate reached the endpoint or completed a traversal. Continued optimization
  from the selected state therefore did not beat the 716.8k image-only
  champion (`0.170687` Dice); the image-only policy remains below the
  preregistered 0.40 Dice/traversal target.

### M15: Supervised GT-mask-input learning curve

- The explicitly supervised seven-channel baseline changes only the policy
  observation by adding a local GT small-bowel mask. It is an annotation-using
  diagnostic and must not be presented as an image-only result.
- At 102,400 frames:
  - mean Dice `0.176511` (pt14 `0.196363`, pt18 `0.156659`);
  - mean endpoint distance `40.819 mm` (55.399, 26.239);
  - zero endpoint reaches and traversals.
- Resuming the exact 102.4k state to 512,000 total frames produced:
  - 153.6k: Dice `0.143622`;
  - 204.8k: Dice `0.073084`;
  - 256.0k: Dice `0.177084`;
  - 307.2k: Dice `0.185085`;
  - 358.4k: Dice `0.165645`;
  - 409.6k: Dice `0.224446`;
  - 460.8k: Dice `0.149742`;
  - 512.0k: Dice **`0.298841`** (pt14 `0.336345`, pt18 `0.261337`).
- At 512k the mean endpoint distance was `33.694 mm` (40.305, 27.083),
  both validation returns were positive (10.326, 22.804), but neither case
  reached the exact 3-mm endpoint or completed a traversal. This is real
  supervised-input progress, not success under the registered criterion.
- A second state-preserving continuation was run from 512k to 1,024,000 total
  frames with validation every 51,200 frames. It completed in 32m13s without
  OOM; peak CUDA memory was 13,406.5 MiB allocated / 14,004.0 MiB reserved.
  The final console summary was approximately `0.06` mean Dice and zero
  success. Per-gate JSON ranking and best-checkpoint trajectory audit remain
  required before selecting or extending this run.

### M16: Joint categorical movement ablation

- Exact per-gate extraction confirmed that the 512k supervised checkpoint is
  the retained champion. The next 51.2k frames already regressed from
  `0.298841` to `0.202396` Dice, and all later gates through 1,024k remained
  between `0.035600` and `0.084947`; no endpoint reach or traversal occurred.
- The 512k trajectory audit identified a deterministic movement failure hidden
  by the aggregate Dice:
  - pt14 visited 307 positions overall but only two positions in its final 512
    steps, with a 100% final-window immediate-reversal rate;
  - pt18 visited 394 positions overall and 141 in its final 512 steps, but its
    final-window immediate-reversal rate was still 48.8%;
  - the stochastic training policy was not globally collapsed (joint maximum
    action probability `0.03269`, logit standard deviation `0.81492`).
- The factorized categorical policy cannot represent correlations between the
  three displacement coordinates. Its deterministic rollout independently
  selected axis modes `(4, -4, 4)` at 512k, which can turn small marginal-logit
  changes into a long diagonal mode and a two-point reversal even while sampled
  training actions remain diverse.
- Queue a one-factor, from-scratch 102.4k supervised-input ablation using the
  existing joint categorical implementation. It assigns a probability to each
  complete nonzero integer displacement, preserves exact PPO likelihoods, and
  initializes equal total mass for each Chebyshev step length. Reward, GT-mask
  input, GRU, split, patch, optimizer, seed, and validation remain unchanged.
- The run completed in 7m20s without OOM. Held-out gates were:
  - 25.6k: Dice `0.041949`, endpoint distance 139.021 mm;
  - 51.2k: Dice `0.039119`, endpoint distance 83.992 mm;
  - 76.8k: Dice `0.058441`, endpoint distance 174.832 mm;
  - 102.4k: Dice `0.046047`, endpoint distance 170.352 mm.
  No endpoint reach or traversal occurred. The 728-way joint head removes the
  independent-axis mode defect but is too diffuse at this budget and is
  rejected relative to the matched factorized 102.4k baseline (`0.176511`).

### M17: Undilated revisitation penalty

- Trajectory audit showed that the active reward had only an opportunity cost
  for revisitation: zero new Dice/cell reward plus the fixed step cost. The old
  binary overlap term was removed because its segment included the mandatory
  starting voxel, making it fire on virtually every valid action.
- Add an explicit agent-owned revisit component:
  - exclude the segment's mandatory starting voxel;
  - evaluate prior occupancy on `cumulative_path_mask_pen`, the undilated
    centerline, before inserting the current segment;
  - penalize the occupied fraction rather than the occupied voxel count;
  - retain the 9-mm Euclidean tube only for the policy feature, Dice metric,
    and path output.
- BOMOPI scale is `0.01`. Numerical contract after a segment has been visited:
  - unseen one- or four-voxel tail: `0`;
  - fully revisited one- or four-voxel tail: `-0.01`;
  - partial overlap: `-0.01 * overlap_fraction`;
  - a two-action forward/backward cycle receives an additional `-0.01`, and
    continued two-position oscillation receives `-0.02` per cycle.
- Dilation remains segment-local: `_add_path_segment` expands only the current
  executed line by the fixed physical-radius offsets and unions those voxels
  into the cumulative mask. It never redilates the accumulated mask.

### M18: Normalized Shin reward contract

- Add an opt-in `reward_contract` so this ablation cannot silently change prior
  experiments. `shin_normalized` implements Algorithm 1 from Shin and Summers
  (MICCAI 2022) after division by `r_val2=6`:
  - new historical-maximum GDT progress: `delta_gdt / theta`;
  - abrupt new maximum: `-1`;
  - mean wall response: `-[0, 1]`;
  - binary revisit and zero movement: `-4/6`;
  - outside-segmentation assignment/overwrite: `-4/6`;
  - terminal scale: `100/6`.
- The cumulative-path overlap check intentionally uses the undilated executed
  centerline tail and excludes the mandatory current voxel. This preserves the
  user's registered revisitation semantics; a 6- or 9-mm cumulative cylinder
  around the previous endpoint would classify ordinary short forward movement
  as a revisit in the present voxel-grid implementation.
- A reproducible pre-training audit is available as:

  `PYTHONPATH=src uv run --no-sync python scripts/audit_navigator_reward_contract.py --contract shin_normalized`

- With the current 6-mm axis limit, `theta=6*sqrt(3)=10.392 mm`. The literal
  normalized reward gives:
  - unseen in-bowel 6-mm progress: `+0.577350`;
  - first 6-mm forward/back cycle: `-0.089316`;
  - established two-position cycle: `-1.333333`;
  - novel in-bowel tangent motion with no detected wall: `0`;
  - a wall response of `0.25`: `-0.25`;
  - outside endpoint, revisit, or zero action: `-0.666667`;
  - abrupt new maximum: `-1`;
  - horizon failure at Dice `0.30`: `-11.666667`.
- Another literal paper-level weakness is delayed failure under discounting. Novel
  in-bowel tangent motion with a zero wall response earns exactly zero, so at
  `gamma=0.999` delaying a Dice-0.30 failure until step 2,048 discounts its
  start-state contribution from `-11.666667` to `-1.504878`, an apparent
  improvement of `+10.161789`.
- The same audit exposed two unacceptable literal failure modes:
  - Algorithm 1 checks only the endpoint segmentation. An inside-to-inside
    cross-loop jump can keep the `+0.577350` progress reward when the
    Meijering wall response is zero or weak.
  - Reaching the endpoint receives a positive terminal reward at any nonzero
    coverage: Dice `0.10 -> +1.666667`, Dice `0.30 -> +5.000000`. This
    conflicts with the registered endpoint-plus-`0.40`-Dice success criterion.
- Add and select `shin_normalized_guarded` for training. It retains the
  normalized dense scales but makes three explicit anti-hacking corrections:
  - any background voxel on the complete executed segment overwrites the step
    with `-0.666667` and cannot advance the historical GDT maximum;
  - endpoint arrival below Dice `0.40` takes the failure terminal branch.
    Hence Dice `0.10 -> -15.000000`, Dice `0.30 -> -11.666667`, and Dice
    `0.40 -> +6.666667`.
  - each valid step costs `(1-gamma)*(100/6)`. At `gamma=0.999`, this is
    `-0.016667`: full 6-mm progress remains `+0.560684`, novel tangent motion
    is `-0.016667`, a first forward/back cycle is `-0.122650`, and established
    oscillation is `-1.366667` per cycle. Delaying the same Dice-0.30 failure
    to step 2,048 now has return `-16.023869`, which is `-4.340536` worse than
    failing immediately.
- The strict evaluation metric remains unchanged. This is an explicitly
  supervised-input/reward diagnostic (GT segmentation policy channel plus GT
  segmentation/GDT reward), never an image-only or annotation-free result.
- Matched planned gate:
  - unit:
    `navigator-bomopi-gru-p32-gtmask-shin-norm-guarded-102k-v1.service`;
  - from scratch, seed 42, factorized categorical GRU, 32-cubed voxel patch
    (48 mm at 1.5-mm spacing), 6-mm axis displacement, 2,048-step horizon;
  - 102,400 frames, validation on only pt14 and pt18;
  - all previous potential/coverage/recovery/distance/step/episodic shaping
    disabled by the launcher for this named contract.
- Pre-launch local checks:
  - guarded and literal numerical audits pass their cycle assertions;
  - all 20 dependency-free reward-helper unit tests pass under `uv`;
  - Python syntax, shell syntax, and `git diff --check` pass;
  - 63 reward/environment integration tests and 19 PPO, annotation-separation,
    metric, and preflight regression tests pass under `uv` on `commander`.
- Launched from scratch at 2026-07-29 19:09 Europe/Sofia:
  - unit:
    `navigator-bomopi-gru-p32-gtmask-shin-norm-guarded-102k-v1.service`;
  - invocation: `e6cde9dc2c754562b30b5ae2b923302f`;
  - TensorBoard:
    `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-p32-gtmask-shin-norm-guarded-102k-v1/data/bomopi_resampled2_unique-v1/tensorboard/20260729-190933-950936`;
  - startup verified the intended 15 training cases and only pt14/pt18 for
    validation, seven policy channels including the explicitly supervised GT
    mask, the guarded reward contract, and zero legacy reward scales;
  - initial CUDA allocation was approximately 13.0 GiB on the 16-GiB 5070 Ti.
- First held-out gate at 25.6k frames:
  - mean Dice `0.120673` (pt14 `0.171466`, pt18 `0.069880`);
  - mean endpoint distance `195.542 mm` (154.558, 236.525);
  - both episodes reached the 2,048-step horizon, with no endpoint reach or
    traversal;
  - the matched prior-reward run had lower Dice (`0.083770`) but better endpoint
    distance (`140.967 mm`) at the same gate. This is only an early coverage
    improvement, not evidence of traversal.

### M19: Historical March-May 2025 implementation audit

- Audit the initial Navigator lineage (`beb605e` through `7fbbe82`) and
  `notebooks/grl_pathtracking.py` against the paper. The strongest conclusion
  is that the failed runs were not controlled verbatim reproductions: several
  independent environment and PPO faults were active at different commits.
- Movement/distribution mismatches were severe and changed repeatedly:
  - `e313bf8` and `4873d2a` sampled a `TanhNormal` action in `[-1,1]`, then the
    environment applied the Beta-style mapping `2*a-1`, producing an asymmetric
    effective range `[-3,1] * max_step`;
  - `dbf7c9d` used `(a+1)*max_step`, so movement was nonnegative on every axis;
  - the mapping became briefly coherent for the voxel-scaled `TanhNormal` in
    `9ebc38e`; but
  - `b0a25f7` sampled an already voxel-scaled `TanhNormal` action and scaled it
    a second time with `(2*a-1)*max_step`, commonly producing enormous
    out-of-bounds jumps.
  The Beta distribution and matching `[0,1] -> [-max,max]` transform were not
  simultaneously coherent until `7fbbe82` on 2025-05-02.
- Reward arithmetic made nearly all ordinary movement bad even in `7fbbe82`:
  - the cumulative path was initialized as a 6-mm sphere;
  - `line_nd` included the mandatory current voxel; therefore
    `cumulative_path_mask[S].sum().bool()` was true on virtually every action;
  - each action consequently paid `-4` revisitation;
  - the code also doubled the paper wall term to `-12*mean(wall)`;
  - a 1.5-mm axial advance at the configured scale earned only about `+0.866`
    GDT before the unavoidable `-4` and wall terms.
  PPO was therefore asked to distinguish degrees of negative return rather
  than receiving the intended positive signal for valid progress.
- GDT/action units were inconsistent:
  - early configurations used only 2-mm maximum axis displacement, which floors
    to one 1.5-mm voxel rather than the paper's 10-mm action;
  - early `theta` formulas used `sqrt(3)*(d/spacing)^2`, squaring the step
    rather than computing `sqrt(3*d^2)`;
  - `7fbbe82` used `theta=sqrt(3*6^2)=10.392` while its GDT was measured in mm
    and a 9-mm-per-axis diagonal can advance 15.588 mm. Legitimate diagonals
    could therefore be classified as abrupt GDT jumps and penalized.
- Path bookkeeping was also unstable:
  - before `b0a25f7`, only endpoints rather than every crossed segment voxel
    were reliably marked;
  - `b0a25f7` then repeatedly dilated the entire accumulated mask after every
    step, causing old visits to grow with episode length rather than expanding
    only the new segment;
  - fixed-name `wall_map.nii`, `gdt_start.nii`, and `gdt_end.nii` caches meant
    corrected code could silently keep using artifacts produced by an older
    broken implementation.
- The claimed paper-like commit `7fbbe82` introduced a PPO minibatch indexing
  bug: its inner loop iterated with `_` but sliced `batch_data[i:i+batch_size]`
  using the outer collector index. Most of every 512-frame rollout was ignored
  while a small overlapping slice was optimized repeatedly.
- The standalone `notebooks/grl_pathtracking.py` did preserve Beta likelihoods
  correctly, so its main problem was not PPO's probability ratio. It still had
  the always-on `-4` revisit term, computed a millimetre-valued GDT against
  voxel-valued `theta=11.547` for a nominal 10-mm action, actually executed at
  most 9 mm because of flooring, terminated immediately on leaving the
  segmentation, and used only a 1M-frame budget with a 1,024-step horizon and
  batch size 128. It therefore was also not the paper's environment/training
  contract despite having the correct distribution family.
- Training exposure and hyperparameters were not matched either:
  - 32,768 episodes were assigned to one subject before switching, whereas the
    paper collected from four scans in parallel;
  - learning rate was `3e-5` rather than `1e-5`, batch size 128 rather than 32,
    maximum action 9 rather than 10 mm, and horizon 1,024 rather than 800.
- Dataset mismatch remains a separate paper-level limitation even after code
  fixes. Shin's wall signal assumed oral-contrast CT with a bright lumen and
  demonstrated that removing wall components reduced performance. The BOMOPI
  Meijering responses are mostly black, so the main signal intended to prevent
  cross-loop shortcuts is absent. Moreover, a segmentation-derived GDT can
  itself shortcut wherever adjacent bowel loops touch or the annotation
  bridges them.
- Overall attribution:
  1. action mapping/distribution mismatches and unconditional revisit penalties
     are sufficient to explain the earliest failures;
  2. path dilation/caching and the later PPO minibatch bug prevented the
     May-2025 "verbatim" version from being a valid reproduction;
  3. even a correct reproduction may not transfer to BOMOPI without a useful
     wall representation or a topology-preserving target because the paper's
     image/annotation assumptions differ.

### M20: Historical June-August 2025 implementation audit

- Audit all Navigator commits dated June through August 2025, with particular
  attention to `068dc4d` ("word for word implementation of the original
  code"), the July orientation/config branches, and the July RND branch.
- The Git topology is important: `068dc4d` is the tip of `origin/og`, forked
  from `eec05e0` through `505adea`. The later `956ff64`, `7af9094`, and
  `ebf5c73` commits are a sibling lineage forked from `eec05e0`; they are not
  descendants of the word-for-word revision. The closest June reproduction
  was therefore abandoned rather than incrementally repaired.
- `068dc4d` plausibly was the June version that "got closer":
  - the Beta policy and `[0,1] -> [-max,+max]` environment mapping were
    coherent, and the independent three-axis Beta likelihood was retained for
    PPO;
  - the earlier minibatch-indexing error was absent: the inner PPO loop sliced
    with its own `j` index;
  - at 1.5-mm data spacing, its nominal 40-voxel patch is 60 mm across, its
    four-voxel cumulative-path radius is 6 mm, and its six-voxel axis action is
    9 mm. These values are much closer to the paper's 60-mm patch, 6-mm
    cylinder, and 10-mm action than the names in the 1-mm `Config` imply;
  - it restored the three intended actor channels: CT, Meijering wall
    response, and cumulative path, and used the paper's mean rather than
    maximum wall response;
  - it restored the paper PPO values for learning rate (`1e-5`), discount
    (`0.99`), clip (`0.2`), entropy (`0.001`), and five epochs.
- It was nevertheless not a valid word-for-word reproduction:
  - minibatch size was 256 rather than 32, the horizon was 1,024 rather than
    800, collection used one environment rather than four mixed PathSet and
    SegmSet environments, and the critic did not receive the paper's optional
    GT-path channel;
  - `theta` was computed as `6*sqrt(3)=10.392` even though the GDT used the
    NIfTI physical spacing. A six-voxel diagonal at 1.5 mm can advance
    15.588 mm, while the paper's 10-mm-per-axis threshold is 17.321 mm.
    Legitimate diagonal progress could therefore become an abrupt-jump
    penalty;
  - the revisit workaround `cumulative_path_mask[S][3:]` was geometrically
    invalid. After a six-voxel straight segment is dilated by four voxels,
    the next straight segment's offsets three and four are already occupied,
    so an ordinary continuation pays `-4`. Conversely, actions containing at
    most three rasterized points have an empty checked tail and can revisit
    without penalty. This shapes action length/direction instead of detecting
    actual return to an old centerline;
  - a nominal clean six-voxel axial advance earns at most
    `6*(9/10.392)=5.196` from physical GDT progress, then can lose `4` to the
    false revisit and up to `6` to the wall term. The reward for a correct
    continuation could therefore still be negative;
  - validation ran ten rollouts with the training reset distribution and chose
    the one with the highest *mean* step reward. It did not perform the paper's
    fixed pylorus-to-end test. Because 60% of resets were reverse-end or
    middle-point starts, the selected result could be a much easier partial
    traversal, and averaging instead of summing favored short terminal
    jackpots;
  - the terminal segment that entered the goal radius was not inserted into
    the path mask/history, and horizon truncation reported zero final coverage.
    The logged coverage was consequently not a stable full-traversal metric.
- The other June lineage added several confounders instead of repairing these
  issues:
  - `956ff64`/`7af9094` reduced the patch to 16 voxels (about 24 mm at the
    relevant spacing), removed the useful wall channel in favor of repeated CT
    patches, took the maximum wall response, disabled revisit, added a
    time-dependent survival term, and used hand-binned terminal coverage;
  - `7af9094` dilated the GT segmentation before computing its Dice denominator
    and allowed region while continuing to use GDT caches from the undilated
    segmentation. This could bridge nearby loops and inflate/alter coverage;
  - `3a0b0b0` introduced an undilated revisit map but checked the entire
    rasterized segment, including its mandatory current voxel. From the second
    action onward, every nonempty segment therefore paid the revisit penalty.
- July continued from `3a0b0b0`/the sibling master line, not from `068dc4d`:
  - the active 16-voxel patch retained only about 24 mm of physical context;
  - the revisit map still included the mandatory current voxel, so every
    action after the first paid `-3` or `-4`;
  - the "survival" reward was
    `gamma*t-(t-1) = 1-(1-gamma)*t`. At `gamma=0.999`, this pays about `+1`
    initially, remains positive until step 1,000, and becomes negative
    thereafter. It encourages delaying termination independently of anatomical
    progress and is not potential-based shaping over an environment state;
  - the July config lineage also added a second GDT potential bonus on top of
    the paper GDT term and used the maximum wall response. These changes can
    make progress look strong numerically while changing the paper objective.
- The July orientation branch (`1968d58`, repaired enough to construct
  TensorDict inputs in `6a19333`) gave the actor its last action as a
  quaternion-like orientation and gave only the critic an explicit
  goal-direction quaternion. The direction encoding used a unit direction as
  a rotation vector of fixed one-radian magnitude rather than a rotation from
  a canonical heading. The branch also:
  - restarted 50% of episodes from previously achieved GDT maxima, making
    training coverage a curriculum/partial-path statistic rather than evidence
    of endpoint-to-endpoint traversal;
  - paid `100*(2*Dice)` on every termination in `1968d58`, including leaving
    the allowed area. At Dice 0.20, crashing received a `+40` terminal bonus;
  - allowed travel in a ten-voxel dilation around the bowel, while the
    Meijering channel was weak. Explicit heading therefore made straight-line
    persistence easier without supplying the missing wall evidence, consistent
    with the commit message that the agent was "back to driving through a
    wall."
- The July RND commit `cc35a0b` did not establish a usable curiosity baseline:
  - intrinsic reward defaulted off;
  - its RND constructor expected a Gymnasium `VectorEnv`, observation spaces,
    and `num_envs`, but was passed the TorchRL bowel environment;
  - training selected `observations`/`next_observations`, while the collector
    stored nested `actor` and `("next","actor")` keys and the RND code itself
    later requested `observation`/`("next","observation")`;
  - the default Mnih 3-D encoder applies an 8-kernel/stride-4 convolution and
    then a 4-kernel convolution, which is invalid for a 16-cubed patch
    (`16 -> 3`, then kernel 4);
  - even after repair, feeding the growing cumulative-path channel to RND would
    reward path-map novelty and could incentivize wandering rather than bowel
    traversal unless anatomy/path novelty were separated and gated.
- No commits of any kind exist between 2025-07-20 and 2026-01-01, and no
  Navigator commits exist in August 2025 on any local or remote-tracking
  branch. There is therefore no August implementation to audit from Git; any
  August run would need external WandB/checkpoint/log artifacts to reconstruct.
- Overall attribution:
  1. `068dc4d` was probably genuinely closer because its physical field of
     view, tube radius, action scale, action distribution, and PPO settings were
     the closest combination to Shin;
  2. its false revisit geometry, wrong physical `theta`, biased validation, and
     data/wall mismatch were still sufficient to prevent reliable complete
     traversal;
  3. later June and July results are not clean continuations of that branch and
     introduced enough reward and metric confounders that they should not be
     used to judge whether the original method could work;
  4. August contributes no repository evidence.

### M21: Guarded-Shin 102.4k result and historical-direction decision

- The supervised-input `shin_normalized_guarded` screening run completed
  successfully in 7m19s with exit status zero and 13,405 MiB peak allocated
  CUDA memory. It did not produce traversal:
  - 25.6k: Dice `0.120673`, endpoint distance `195.542 mm`;
  - 51.2k: Dice `0.119211`, endpoint distance `152.992 mm`;
  - 76.8k: Dice `0.109110`, endpoint distance `125.960 mm`;
  - 102.4k: Dice `0.104079`, endpoint distance `124.816 mm`;
  - both pt14 and pt18 used all 2,048 steps at every gate, with zero endpoint
    reaches and zero traversal successes.
- The opposite Dice/distance trends suggest some global displacement toward
  the endpoint without correct bowel coverage. Final path statistics confirm
  strong loitering:
  - pt14: 84 unique positions among 2,049, only 27 unique in the last 512, and
    28.2% immediate reversals in that final window;
  - pt18: 127 unique positions among 2,049 and only 15 unique in the last 512;
  - neither trajectory used zero actions, so this is cycling/limited-state
    motion rather than rounding to a stationary action.
- The final training batch had no positive GDT signal:
  - `train/reward_gdt=0`, `train/max_reward=-0.666667`;
  - background/invalid and off-target components accounted for almost the
    entire mean reward (`-0.327474` and `-0.339193`);
  - mean training reward was `-0.679271`, while the policy remained stochastic
    (`max_action_probability=0.01683`).
  This is a reward-regime collapse, not merely insufficient training time.
- Direct comparison:
  - the matched modern 102.4k baseline reached Dice `0.176511` and endpoint
    distance `40.819 mm`;
  - the explicit undilated-revisit 102.4k run reached Dice `0.249127` and
    endpoint distance `68.339 mm`;
  - the same modern lineage peaked at 512k with Dice `0.298841` and endpoint
    distance `33.694 mm`, but continued training collapsed to Dice `0.062772`
    by 1.024M;
  - none reached the endpoint. The modern implementation has demonstrated
    nontrivial partial tracking, but neither reward family is stable enough for
    an unchanged multi-million-step run.
- Decision:
  - do not continue the completed guarded-Shin checkpoint unchanged;
  - do not check out and resume historical `068dc4d`, whose revisit geometry,
    physical theta, validation, and bookkeeping are invalid;
  - retain the current tested movement, PPO likelihood, metric, recurrent-state,
    logging, and annotation-separation infrastructure, but construct one
    controlled "068-repaired" ablation from the historical design:
    60-mm physical field of view, approximately 9-10-mm action, 6-mm path
    radius, CT/wall/undilated-path actor inputs, physical GDT/theta, current-voxel
    exclusion for revisit, fixed endpoint-start validation, and strict
    endpoint-plus-Dice success;
  - remove the July survival/potential/terminal-crash shaping and do not treat
    GT-mask-policy results as annotation-free evidence.
- Proposed bounded go/no-go criterion: screen at 256k, run to 512k only if
  valid/GDT-positive action rates and final-window path diversity are improving,
  and require by 1M at least one held-out endpoint reach plus mean Dice near the
  registered `0.40` target. If the repaired historical geometry still fails,
  stop extending PPO budgets and pivot the main method to the proposed
  energy/spline or hybrid tracker.

### M22: Repaired-068 implementation and preregistered 256k screen

- Implemented `shin_normalized_repaired` inside the current Navigator rather
  than restoring the historical branch. This retains the tested movement,
  recurrent PPO likelihood, full terminal-segment bookkeeping, component
  telemetry, and fixed start-to-end validation.
- The repaired reward keeps normalized Shin new-maximum GDT, mean wall
  response, and binary revisit on the prior undilated centerline tail. It:
  - computes the GDT jump threshold from the executable physical diagonal:
    `6 vox * 1.5 mm * sqrt(3) = 15.588 mm`;
  - gates all positive GDT when an executed segment crosses the
    endpoint-connected target boundary;
  - replaces the guarded run's flat `-0.6667` off-target cliff with
    `-(2/3) * min(max_segment_target_distance / 15.588 mm, 1)`;
  - retains the discount-consistent per-step cost
    `-(1-gamma)*(100/6)`, which is `-0.1667` at `gamma=0.99`;
  - permits positive terminal reward only when both the endpoint and the
    registered Dice threshold are reached.
- Numerical failure-mode audit at the registered `gamma=0.99`, 800-step
  horizon, and 15.588-mm physical diagonal:
  - a new 6-mm inside-target advance returns `+0.2182` before wall response
    (the registered 9-mm axial advance returns `+0.4107`);
  - because the paper wall term is subtractive, a mean normalized wall
    response above `0.4107` would make even that 9-mm axial advance negative;
    wall-component telemetry must therefore be a go/no-go signal rather than
    assuming the cached Meijering map is informative;
  - a novel tangent move returns `-0.1667`;
  - an established revisit returns `-0.8333`;
  - a cross-loop action through a 1.5-mm background gap returns `-0.2308`
    and receives zero GDT credit;
  - a full-action-distance excursion returns up to `-0.8333`;
  - forward/backward and established two-position cycles return `-0.4226`
    and `-1.6667`, respectively, using the registered 9-mm advance;
  - endpoint arrival at Dice `0.10` or `0.30` receives terminal components
    `-15.0` and `-11.6667`; Dice `0.40` receives `+6.6667`;
  - discounted zero-wall wandering to the horizon is `-4.8318` worse than
    immediate failure, so delaying failure is not profitable.
- Added `shin_068_repaired` policy observations: exactly normalized current CT,
  the original wall response, and a separate undilated agent-owned centerline
  map. The policy receives only previous movement direction as scalar context;
  it receives no GT mask, GDT, goal direction/distance, absolute position, or
  time fraction. The actor path includes the initial seed, while the separate
  revisit ledger deliberately begins empty so the mandatory start is not a
  false revisit.
- Added the reproducible
  `scripts/run_navigator_bomopi_068_repaired.sh` launcher:
  60-mm (`40^3`) patch, 9-mm action (`6` vox), 6-mm path radius (`4` vox),
  GRU plus factorized categorical action likelihood, `lr=1e-5`, `gamma=0.99`,
  entropy `0.001`, true recurrent minibatch/sequence length `32`, five PPO
  epochs, 800-step horizon, and a
  bounded 256k-frame screen.
- The first 4k CUDA smoke with the paper's minibatch `32` failed during the
  first GroupNorm forward pass: PyTorch held 10.42 GiB and requested another
  3.91 GiB with only 3.84 GiB free on the 15.50-GiB GPU. This was a real
  capacity miss rather than substantial allocator fragmentation (318 MiB was
  reserved but unallocated). The initial hypothesis was that PPO minibatch
  activation memory caused the failure, so a second smoke requested `16`.
- A second smoke with advertised minibatch `16` failed with the exact same
  allocation. The complete traceback locates the allocation at
  `adv_module(batch_data)`, before PPO minibatching. The original recurrent
  sampler also makes any requested minibatch below the configured 64-step
  recurrent sequence into one effective 64-transition minibatch. The corrected
  hardware setting therefore restores minibatch `32`, sets recurrent sequence
  length `32`, and reduces only the rollout-wide GAE chunk from 1,024 to 512
  frames. Five PPO epochs and every task parameter remain unchanged.
- The corrected third 4,096-frame CUDA smoke completed successfully in 31.6 s
  (about 199 frames/s) with 7,532.7 MiB peak allocated and 9,236.0 MiB peak
  reserved CUDA memory. It completed all five PPO epochs with final approximate
  KL `0.001341`, confirming the 512-frame GAE chunk has substantial memory
  headroom without clipping the registered PPO update count.
- Final random-policy-scale reward components from that smoke were:
  total mean `-1.0201`, off-target `-0.5630`, wall `-0.1782`, step `-0.1615`,
  revisit `-0.0638`, invalid `-0.0208`, terminal `-0.0317`, and GDT
  `-0.0011`. Off-target distance is initially the largest term, as expected,
  but it is no longer the guarded contract's all-or-nothing `-0.6667` value;
  the wall mean remains below the `0.4107` threshold that would overwhelm a
  full 9-mm axial progress reward.
- Started the preregistered screen as
  `navigator-bomopi-gru-068-repaired-256k-v1.service`. Its TensorBoard run is
  `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-068-repaired-256k-v1/data/bomopi_resampled2_unique-v1/tensorboard/20260729-211757-953642`.
- Verification in an isolated copy on the Linux CUDA host, using its existing
  `uv` environment:
  - shell syntax and reward audit passed;
  - focused reward/environment suite: 65 tests passed;
  - complete `test_navigator_*.py` suite: 95 tests passed.
- Preregistered gate remains unchanged: continue to 512k only if positive-GDT
  action rate and final-window path diversity improve; by 1M require at least
  one held-out endpoint reach and mean Dice near `0.40`, otherwise pivot away
  from a pure PPO primary method.

### M23: Repaired-068 256k failure and exact boundary-safe policy

- The completed `navigator-bomopi-gru-068-repaired-256k-v1` run was
  technically stable but failed its scientific screen:
  - 40,960 frames: held-out mean Dice `0.011352`, endpoint distance
    `223.939 mm`;
  - 81,920 frames: held-out mean Dice `0.014162`, endpoint distance
    `203.067 mm`;
  - both validations ran 800 steps with zero endpoint reaches and zero
    traversal successes;
  - the final 256k training batch had mean Dice `0.00966`, invalid-action
    reward `-0.658854`, off-target reward `-0.007813`, zero GDT reward, and
    maximum per-step reward `-0.666667`.
- Deterministic path inspection exposed a boundary exploit, not merely slow
  learning. Each validation path made ten moves to 11 positions and then
  selected outward actions for roughly 790 steps. A rejected action paid the
  flat invalid penalty `-2/3`, whereas executing an off-target action could
  pay step, distance, and wall costs in addition. The policy therefore found
  a locally preferable stationary boundary action whose categorical
  probability still participated in PPO.
- Replaced that distribution with exact state-dependent joint masking:

  `pi(a|s) = exp(z_a) / sum(exp(z_b), b in A(s))` for `a in A(s)`, and zero
  otherwise.

  `A(s)` contains every configured nonzero displacement whose endpoint remains
  inside the volume. The mask depends only on image bounds and the current
  position; it does not inspect the bowel mask, GDT, goal, reward, or any other
  label. The action and its masked log-probability are stored together in the
  rollout, so PPO evaluates exactly the distribution that sampled the action.
  The environment executes the selected integer displacement directly and
  raises if the mask/action contract is ever violated.
- Added collapse telemetry to both training and deterministic validation:
  executed/invalid action fraction, positive-GDT fraction, off-target fraction,
  recent unique-position fraction over 256 steps, immediate reversals,
  boundary-state fraction, and the fraction of actions available in the
  current mask.
- Fixed a separate scheduling defect. Validation and regular checkpoints used
  `num_updates % interval == 0`; variable PPO epoch counts/KL early stopping
  could jump over a multiple and silently omit the event. Scheduling now fires
  whenever an interval threshold is crossed and advances the threshold past
  the current counter. This explains why the 256k run produced only two
  validations despite a nominal 400-update interval.
- Verification on the Linux CUDA host used the existing project environment
  through `uv --no-sync`:
  - focused reward/environment/PPO suite: 81 tests passed;
  - complete `test_navigator_*.py` suite: 99 tests passed;
  - the recurrent PPO test confirms every sampled action is feasible, the
    stored log-probability equals an independently reconstructed exact masked
    log-probability, and the PPO backward pass remains finite.
- Registered follow-up: a 4,096-frame CUDA smoke, followed only on technical
  success by `navigator-bomopi-gru-068-masked-64k-v1`. The 64k screen validates
  approximately every 16,384 frames and must maintain exactly zero invalid
  actions. Continue beyond 64k only if held-out Dice/endpoint distance,
  positive-GDT rate, and path diversity show a coherent improvement without a
  boundary or short-cycle collapse.
- The 4,096-frame CUDA smoke completed in about 20 seconds of training at
  roughly 200 frames/s. Peak CUDA allocation/reservation was
  `7,551.998/9,214 MiB`, leaving adequate headroom on the 15.5-GiB GPU.
  Every training batch and both held-out rollouts had executed-action fraction
  `1.0` and invalid-action fraction `0.0`, verifying that the old stationary
  invalid-action exploit is unreachable. PPO remained finite; final
  approximate KL was `0.004834`.
- The smoke is not evidence of tracking quality. Its 4k held-out mean Dice was
  `0.009305`, endpoint distance `247.666 mm`, positive-GDT fraction `0.010625`,
  and recent unique-position fraction `0.033203`. Although every action
  executed, deterministic trajectories already spent `95.69%` of states near
  an image boundary. The 64k screen must therefore distinguish a temporary
  untrained mode from a new boundary-following collapse.
- `navigator-bomopi-gru-068-masked-64k-v1.service` completed all 65,536 frames
  with exit status zero in 5m59s. Peak CUDA allocation/reservation was
  `7,669.6/9,858.0 MiB`. TensorBoard:
  `/home/matey/project/segmentor/checkpoints/navigator-bomopi-gru-068-masked-64k-v1/data/bomopi_resampled2_unique-v1/tensorboard/20260730-002703-956076`.
- Held-out two-case deterministic gates:
  - 16,896 frames: Dice `0.001097`, endpoint distance `197.590 mm`,
    positive-GDT fraction `0`, recent unique-position fraction `0.046875`,
    boundary-state fraction `0.94625`;
  - 33,280 frames: Dice `0.006174`, endpoint distance `255.989 mm`,
    positive-GDT fraction `0.0075`, recent unique-position fraction `0.007813`,
    boundary-state fraction `0.975`;
  - 49,664 frames: Dice `0.006110`, endpoint distance `261.558 mm`,
    positive-GDT fraction `0.009375`, recent unique-position fraction
    `0.027344`, boundary-state fraction `0.97`;
  - a separate evaluation of the saved 65,536-frame final policy: Dice
    `0.009334`, endpoint distance `240.737 mm`, positive-GDT fraction
    `0.01625`, recent unique-position fraction `0.019531`, boundary-state
    fraction `0.965625`.
  Every gate retained action-executed fraction `1.0` and had zero endpoint
  reaches and traversal successes.
- Interpretation: exact action masking eliminated the invalid-action reward
  exploit, but did not rescue the repaired Shin state/reward. The
  deterministic policy replaced a stationary outward action with a tiny
  executable cycle along the image boundary. The stochastic training batches
  remained diverse, while deterministic validation collapsed, so continued
  optimization of this unchanged objective is not justified by the 64k
  evidence. Do not extend this checkpoint to 512k or 1M.
- The run contained 639 PPO epochs rather than the nominal 640 because KL early
  stopping is variable. Consequently the next periodic update threshold fell
  just beyond the finite run and the original job emitted only three
  validations. Added a mandatory end-of-run validation unless the exact final
  frame was already scored; this also participates in best-checkpoint
  selection. The final-policy metrics above were recovered with a separate
  eval-only run because the training job predated this fix. Full Linux `uv`
  Navigator suite after the scheduling change: 100 tests passed.
- Final optimization diagnostics show that the boundary mode was selected from
  a still-diffuse policy rather than a nearly deterministic learned
  preference. Maximum action probability was only `0.01215`; the
  entropy-loss magnitude `0.007162` at coefficient `0.001` implies about
  `7.162` nats of raw entropy, or roughly 1,290 effective actions out of 2,196.
  Logit standard deviation was `0.7093`, KL `0.00301`, policy loss `-0.0141`,
  and value loss remained `6.4059`. The final training batch had zero GDT
  reward, mean off-target reward `-0.4510`, revisit `-0.0768`, wall `-0.1277`,
  and total mean reward `-0.8223`.
- Recommended next controlled ablation: retain exact bounds-only masking and
  the repaired metrics, but reduce the joint support from every integer vector
  in the `13^3` cube to physically interpretable direction/length actions
  (26 lattice directions times six step lengths, at most 156 categories).
  This tests whether PPO can assign useful likelihood mass without changing
  reward scales, observation channels, GT separation, or movement semantics.

### M24: Preregistered compact masked-action screen

- Implemented `categorical_action_support=direction_length` as an explicit
  alternative to the unchanged `dense` default. At the registered six-voxel
  maximum it contains the Cartesian product of 26 nonzero directions in
  `{-1,0,1}^3` and lengths 1 through 6: exactly 156 unique integer actions.
  Each selected vector is still executed exactly and the policy is still
  renormalized over the bounds-only feasible subset. No reward, observation,
  label, movement, PPO, validation, or success setting changes.
- Motivation is numerical rather than cosmetic:
  - the dense head's equal-length prior has entropy `7.1830` nats, equivalent
    to about 1,317 uniformly likely actions; its trained 64k entropy was still
    about `7.162` nats, so it had barely reduced action uncertainty;
  - compact support starts at `log(156)=5.0499` nats and removes 524,280 head
    parameters, reducing the registered model from 1,162,629 to approximately
    638,349 trainable parameters;
  - all six axial, face-diagonal, and body-diagonal step scales remain
    available, so the repaired Shin reward magnitudes and physical maximum
    progress threshold are unchanged;
  - a 300,000-point spherical audit estimates the worst nearest-heading error
    of the 26-direction set at `27.56 degrees`. This angular quantization is the
    deliberate price of the ablation and must not be hidden.
- Reproducible launcher:
  `scripts/run_navigator_bomopi_068_masked_compact_64k.sh`. Run a 4,096-frame
  CUDA smoke first, then a fresh 65,536-frame screen only if invalid-action
  fraction remains exactly zero and PPO/CUDA values are finite.
- Preregistered 64k continuation gate versus the failed dense run:
  - final boundary-state fraction below `0.80`;
  - recent unique-position fraction above `0.10`;
  - positive-GDT fraction at least `0.02`;
  - either mean Dice above `0.02` or mean endpoint distance below
    `197.59 mm`, the best dense-screen endpoint gate.
  Require at least three of these four criteria plus a non-collapsing trend to
  justify 256k. The registered long-run target remains at least `0.40` Dice
  with full endpoint-to-endpoint traversal; compact support is not allowed to
  redefine success.
- The managed 4,096-frame CUDA smoke completed with exit status zero in 39s,
  including mandatory final validation. The compact model has the predicted
  638,349 parameters. Peak CUDA allocation/reservation was
  `7,505.0/8,678.0 MiB`; invalid-action fraction was exactly zero and final KL
  was `0.00258`.
- Smoke held-out metrics were Dice `0.012346`, endpoint distance `218.394 mm`,
  positive-GDT fraction `0.006875`, recent unique-position fraction
  `0.035156`, and boundary-state fraction `0.521875`. This is directionally
  better than the dense 4k smoke (`0.009305`, `247.666 mm`, and `0.956875`
  boundary residence), but the two cases were asymmetric: pt14 boundary
  residence was `0.10625`, while pt18 remained at `0.9375`. The smoke passes
  the technical gate for a fresh 64k screen but is not scientific evidence of
  traversal.
- `navigator-bomopi-gru-068-masked-compact-64k-v1.service` completed with exit
  status zero in 6m04s and mandatory final validation at exactly 65,536
  frames. Peak CUDA allocation/reservation was `7,588/9,460 MiB`. All gates
  retained action-executed fraction `1.0`:
  - 16,384: Dice `0.010002`, endpoint `161.032 mm`, positive GDT `0.015625`,
    recent diversity `0.226563`, boundary residence `0.424375`;
  - 32,768: Dice `0.010374`, endpoint `189.193 mm`, positive GDT `0.001875`,
    recent diversity `0.347656`, boundary residence `0.315625`;
  - 49,152: Dice `0.023095`, endpoint `91.552 mm`, positive GDT `0.00125`,
    recent diversity `0.03125`, boundary residence `0.381875`;
  - 65,536: Dice `0.011535`, endpoint `91.446 mm`, positive GDT `0.0025`,
    recent diversity `0.169922`, boundary residence `0.415`.
  No gate reached an endpoint or completed traversal.
- The final gate passes three of four preregistered continuation criteria:
  boundary residence, recent diversity, and endpoint distance. Positive GDT
  fails by almost an order of magnitude, Dice did not retain its 49k peak, and
  failures remain case-specific (pt18 ended at `82.9%` boundary residence and
  `1.95%` recent diversity). This justifies only the registered 256k
  continuation, not a claim of anatomical tracking.
- Final policy diagnostics remain weak: maximum action probability `0.01674`,
  raw entropy approximately `4.9755` nats versus the uniform compact maximum
  `5.0499`, value loss `6.689`, and final training GDT reward `0.000503`.
  Compact support improved deterministic geometry without demonstrating that
  PPO has concentrated on reward-aligned actions.
- The reproducible continuation launcher
  `scripts/run_navigator_bomopi_068_masked_compact_256k.sh` resumes the final
  64k optimizer/policy state and explicitly fixes
  `lr_anneal_timesteps=65536`. The saved scheduler is already at
  `last_epoch=T_max=640` and learning rate `5e-6`; this prevents a longer
  `total_timesteps` value from silently making cosine annealing rise after
  resume. Continue beyond 256k only with a held-out positive-GDT increase and
  sustained Dice/diversity, and never without an endpoint reach by 1M.
- `navigator-bomopi-gru-068-masked-compact-256k-v1.service` resumed exactly
  from 65,536 frames/640 PPO epochs, held learning rate at `5e-6`, and
  completed 256,000 frames with exit status zero in 17m25s. Mandatory final
  validation ran at frame 256,000; peak CUDA allocation/reservation was
  `7,590.5/12,450 MiB`.
- The continuation was highly non-monotonic:
  - its best checkpoint was 81,920 frames: Dice `0.069358`, endpoint
    `186.960 mm`, positive GDT `0.011875`, diversity `0.134766`, boundary
    residence `0.003125`;
  - intermediate Dice then ranged from `0.010686` to `0.039336`, with endpoint
    distance ranging from `42.5` to `240.3 mm` and boundary residence from
    `0.0012` to `0.9019`;
  - the 256,000-frame final policy had Dice `0.036279`, endpoint
    `92.573 mm`, positive GDT `0.0225`, diversity `0.011719`, and boundary
    residence `0.45375`. Neither case reached its endpoint at any gate.
- The final policy's two paths each occupied only three unique positions in
  the last 256 steps. The reproducibly re-evaluated best checkpoint also
  failed end-to-end: pt14 had 157 unique positions over 801 states and 52 in
  the last 256; pt18 had 271 overall but only 17 in the last 256. Its Dice
  (`0.09263`/`0.04608`) therefore reflects the accumulated dilated partial
  route before terminal cycling, not traversal.
- Optimization remained diffuse. At the best 81,920 checkpoint, maximum action
  probability was `0.01361`, logit standard deviation `0.2863`, and raw
  entropy approximately `4.9538` of the `5.0499`-nat maximum. At 256k,
  maximum probability was still `0.01789`, raw entropy `4.9420`, value loss
  had worsened to `8.155`, and invalid actions remained exactly zero.
  Compact support creates better candidate modes, but PPO with this entropy
  scale does not stabilize them.
- Decision: do not extend this run unchanged to 1M. It missed the required
  endpoint reach and remains far below the `0.40` Dice target despite a
  transient `0.069` best checkpoint.

### M25: Preregistered compact categorical entropy-scale ablation

- The `0.001` coefficient from the continuous-action reference is not
  dimensionless with respect to a 156-way categorical. At the best compact
  checkpoint its entropy loss was `-0.004954`, about 34% of the magnitude of
  policy loss `-0.014636`; at 256k it was `-0.004942` versus policy loss
  `-0.008491`. The policy retained roughly 142 effective actions and its
  deterministic mode changed drastically between validation gates.
- Run one fresh, otherwise identical compact 64k screen with
  `ent_coef=0.0001`. This makes the initial maximum entropy contribution about
  `0.000505`, an order of magnitude smaller, without removing stochastic PPO
  exploration. It changes no reward, input, data split, mask, action support,
  likelihood, movement, horizon, or success metric.
- Reproducible launcher:
  `scripts/run_navigator_bomopi_068_masked_compact_lowent_64k.sh`.
  A 4,096-frame smoke must retain finite KL and zero invalid actions before the
  64k run.
- Desired evidence is stable concentration, not merely a sharper bad mode:
  maximum action probability should rise above `0.03` while raw entropy stays
  above `3.0` nats; recent diversity should remain above `0.10`; boundary
  residence below `0.80`; positive-GDT fraction at least `0.02`; and at least
  two consecutive held-out gates should improve Dice or endpoint distance.
  Reject immediately as premature collapse if maximum action probability
  exceeds `0.25` alongside low diversity or boundary cycling. The endpoint
  and `0.40` Dice definitions remain unchanged.
- The managed 4,096-frame smoke completed with exit status zero in 39s.
  Invalid actions remained exactly zero, KL was `0.001731`, and the entropy
  loss fell as intended from about `0.005` to `0.000499`. Maximum action
  probability was only `0.01211`, so reduced entropy pressure did not produce
  premature concentration.
- Smoke validation was scientifically poor—Dice `0.008700`, endpoint
  `159.973 mm`, positive GDT `0.003125`, diversity `0.013672`, and boundary
  residence `0.84`—but an untrained deterministic mode is not the registered
  comparison. The smoke passes only the technical gate for the fresh 64k
  screen.
- `navigator-bomopi-gru-068-masked-compact-lowent-64k-v1.service` completed
  successfully in 6m03s, including mandatory validation at exactly 65,536
  frames. Peak CUDA allocation/reservation was `7,588/9,438 MiB`; invalid
  actions remained exactly zero at every training update.
- Held-out validation did not stabilize:
  - 16,384: Dice `0.022173`, endpoint `232.914 mm`, positive GDT `0.0075`,
    recent diversity `0.082031`, boundary residence `0.916875`;
  - 32,768: Dice `0.006180`, endpoint `185.866 mm`, positive GDT `0.00625`,
    recent diversity `0.011719`, boundary residence `0.955625`;
  - 49,152: Dice `0.013445`, endpoint `218.685 mm`, positive GDT `0.010625`,
    recent diversity `0.035156`, boundary residence `0.899375`;
  - 65,536: Dice `0.011327`, endpoint `169.725 mm`, positive GDT `0.010625`,
    recent diversity `0.195312`, boundary residence `0.441875`.
  No gate reached an endpoint or completed traversal. No single metric
  improved over two successive intervals: Dice improved only from 32k to 49k,
  while endpoint distance improved from 16k to 32k and 49k to 64k.
- Lower entropy regularization did not measurably concentrate the policy.
  Final maximum action probability was `0.01582`; final raw entropy was about
  `4.9784` nats (`-0.00049784 / 0.0001`), equivalent to roughly 145 effective
  actions and essentially unchanged from the ordinary compact run's
  `4.9755` nats. Final logit standard deviation was `0.2826`, KL `0.00472`,
  policy loss `-0.00158`, and value loss `6.966`.
- The final apparent diversity recovery is not anatomical success. Pt14 had
  Dice `0.001722`, zero positive-GDT steps, endpoint distance `107.35 mm`, and
  boundary residence `0.00125`; pt18 had Dice `0.020932`, positive GDT
  `0.02125`, endpoint distance `232.10 mm`, and boundary residence `0.8825`.
  Both exhausted the 800-step horizon and both missed the endpoint.
- Decision: reject this ablation and do not extend it to 256k or 1M. Reducing
  the entropy coefficient by tenfold changed the loss scale as intended but
  neither sharpened the categorical policy nor produced stable held-out
  reward alignment. Further runs should change a learning bottleneck with a
  testable mechanism rather than continue tuning entropy on this objective.

### M26: Exact task/action-signal audit and coverage-dominant repair

- Added a reproducible BOMOPI audit that evaluates the exact
  `shin_068_repaired` observation, 156-action compact masked support, reward,
  path geometry, and horizon:
  `scripts/audit_navigator_bomopi_action_signal.py`. It samples all feasible
  actions at oracle-route states, executes exact oracle rollouts, and compares
  the shortest endpoint route against a skeleton-covering route. The reusable
  route compressor permits only supported on-mask actions and bounded
  route-order lookahead, preventing folded-loop shortcuts.
- The current 800-step repaired-Shin task fails before policy optimization:
  - pt14/pt18 need 1,527/1,811 compact actions for the constructive covering
    route, versus the registered horizon of 800;
  - the 800-action prefixes reach Dice `0.342907/0.295385` and remain
    `120.22/82.17 mm` from the endpoint;
  - exact prefix returns are `-446.41/-384.04`, with only `1.88%/2.63%`
    positive-reward actions;
  - across 128 sampled states the correct route action is positive only
    `3.91%` of the time and top-five by reward only `21.09%` of the time;
  - required revisits are never top-five and average `-0.9481`: the binary
    `-0.6667` revisit, `-0.1667` step cost, and mean wall cost overwhelm
    average GDT credit of only `+0.00036`.
- Exact metric geometry also rejects endpoint-only optimization. The direct
  routes reach the endpoint in 36/54 compact actions but achieve only
  `0.04176/0.03504` Dice. The full covering routes achieve
  `0.53952/0.52583` Dice and both endpoints, so the registered 0.40 target is
  geometrically achievable when the horizon is raised to 2,048.
- Audited a coverage-dominant potential candidate under the same three policy
  channels and compact actions:
  - cumulative Dice scale `500`, max-step-normalized GDT scale `0.1`;
  - target recovery `0.2`, 60-mm graded target-distance cost, step `0.01`,
    undilated revisit `0.01`, no wall or episodic novelty reward;
  - 2,048-step horizon and unchanged endpoint-plus-0.40-Dice success.
- The candidate passes the numerical anti-hacking gate:
  - exact full-oracle returns are `+299.53/+288.57`, discounted returns at
    gamma 0.99 are `+35.19/+20.65`, and both traversals complete;
  - novel oracle actions are positive in `92%` of sampled states;
  - required backtracks average only `-0.0112`, preventing free revisitation
    without making branch return catastrophically bad;
  - all sampled reward-maximizing actions stay fully on target;
  - the direct endpoint shortcuts earn only `+21.04/+17.96`, do not terminate,
    and remain far below the full-route return.
  Cumulative Dice cannot be farmed by cycling because the path tube only
  grows: revisiting adds zero coverage potential and still pays step/revisit
  cost. Positive coverage/GDT are gated off for any segment that leaves the
  endpoint-connected target, while distance/recovery terms correct off-target
  motion.
- Reproducible training launcher:
  `scripts/run_navigator_bomopi_compact_cov500_64k.sh`. Run a 4,096-frame CUDA
  smoke first. Promote to a fresh 65,536-frame screen only with finite PPO,
  exactly zero invalid actions, correct logged scales, and mandatory
  two-case final validation. Continue beyond 64k only if held-out mean Dice
  exceeds `0.05` or endpoint distance falls below `150 mm`, while recent
  diversity stays above `0.10`, boundary residence below `0.80`, and no
  deterministic short-cycle collapse appears. The scientific target remains
  at least 0.40 Dice plus endpoint-to-endpoint traversal on both cases.
- Prelaunch verification on the Linux `uv` environment: complete Navigator
  suite `103 passed`, four subtests passed, and shell syntax passed.
- Managed CUDA smoke
  `navigator-bomopi-gru-compact-cov500-smoke4k-v1.service` completed 4,096
  frames plus mandatory two-case 2,048-step validation in 45s. Effective
  configuration logs exactly record potential coverage `500`, GDT `0.1`,
  recovery `0.2`, target-distance radius `60 mm`, episodic scale `0`, compact
  masked actions, and the 2,048-step horizon. Peak CUDA
  allocation/reservation was `9,394.4/14,550 MiB`.
- Technical telemetry passes promotion: invalid-action fraction was exactly
  zero, maximum observed KL was `0.01069`, final value loss was `0.2767`, and
  the stochastic training recent-position diversity averaged `0.9880`.
  Coverage shaping averaged `+0.004839` per step while the dominant random
  policy cost was the graded target distance at `-0.06817`; no hidden
  terminal, wall, off-target, or episodic reward fired.
- Smoke validation is not learned progress: mean Dice `0.004475`, endpoint
  distance `187.785 mm`, diversity `0.05078`, and zero endpoint/traversal
  success. Pt14 boundary residence was `0.96973` while pt18 was `0.00293`, so
  the fresh 64k screen must establish whether this is an untrained mode or
  another deterministic boundary failure.
- `navigator-bomopi-gru-compact-cov500-64k-v1.service` completed successfully
  in 6m27s with `9,389.3/10,612 MiB` peak CUDA allocation/reservation.
  Deterministic held-out gates were:
  - 16,384: Dice `0.011853`, endpoint `236.90 mm`, diversity `0.02734`,
    boundary residence `0.85254`;
  - 32,768: Dice `0.017330`, endpoint `264.56 mm`, diversity `0.08203`,
    boundary residence `0.98535`;
  - 49,152: Dice `0.001874`, endpoint `157.85 mm`, diversity `0.10352`,
    boundary residence `0.50`;
  - 65,536: Dice `0.031391`, endpoint `228.95 mm`, diversity `0.10742`,
    boundary residence `0.88135`.
  Every gate executed all actions but reached zero endpoints and completed
  zero traversals. The deterministic result misses both preregistered
  continuation alternatives.
- Training did not share the deterministic boundary collapse. Across the
  complete screen, stochastic recent-position diversity averaged `0.9901`,
  boundary residence `0.1341`, coverage reward `+0.01434`, and invalid actions
  exactly zero. Final maximum action probability was only `0.01948`; raw
  entropy remained about `4.943` of `5.050` nats. The entropy-loss magnitude
  remained roughly one quarter of the actor-loss magnitude.
- A separately named stochastic diagnostic used the unchanged final policy,
  exactly pt14/pt18, and three preregistered independent seeds (`101`, `202`,
  `303`), with no checkpoint selection or optimizer updates. Across six
  episodes:
  - mean Dice `0.105487`;
  - mean endpoint distance `118.598 mm`;
  - recent diversity `0.99414` and boundary residence `0.12004`;
  - zero endpoint reaches and zero traversals.
  Per-seed mean Dice was approximately `0.1032`, `0.1242`, and `0.0890`.
  This passes both numerical continuation alternatives and establishes that
  the corrected reward improves expected stochastic behavior, while the
  categorical argmax remains a poor representative of the learned policy.
- Register one state-preserving concentration test:
  `scripts/run_navigator_bomopi_compact_cov500_lowent_256k.sh`. Resume the
  exact 64k policy, critic, optimizer, and scheduler; freeze learning rate at
  its existing `5e-6` floor; change only entropy coefficient
  `0.001 -> 0.0001`; and continue to 256,000 total frames. Validate only near
  128k, 192k, and the mandatory 256k final state. The earlier low-entropy
  ablation used the rejected reward and does not answer this
  stronger-advantage setting.
- Acceptance requires deterministic mean Dice above `0.05` or endpoint
  distance below `150 mm` without boundary/short-cycle collapse, and at least
  one held-out endpoint reach by 256k. Also repeat the fixed three-seed
  stochastic diagnostic at the final checkpoint: it must retain mean Dice
  above `0.10` while improving endpoint distance. Otherwise stop rather than
  extend to 1M.
- `navigator-bomopi-gru-compact-cov500-lowent-256k-v1.service` resumed exactly
  at 65,536 frames / 640 updates, held learning rate at `5e-6`, and completed
  256,000 total frames in 16m41s. Deterministic gates:
  - 131,072: Dice `0.037043`, endpoint `224.15 mm`, diversity `0.09570`,
    boundary residence `0.17407`;
  - 196,608: Dice `0.036540`, endpoint `192.80 mm`, diversity `0.02930`,
    boundary residence `0.00391`;
  - 256,000: Dice `0.070075`, endpoint `105.47 mm`, diversity `0.11328`,
    boundary residence `0.00073`.
  No gate reached an endpoint or completed a traversal. Final progress was
  highly case-asymmetric: pt14 reached Dice `0.13802` and `44.92 mm` endpoint
  error, while pt18 remained at Dice `0.00213` and `166.01 mm`.
- Lower entropy eventually produced only modest concentration. Final maximum
  action probability was `0.02867`, raw entropy about `4.876` nats, and
  invalid actions remained zero. Stochastic training coverage reward improved
  to a final-ten mean `+0.04232` while target-distance cost improved to
  `-0.01377`; value loss remained controlled at `0.216`.
- The fixed final stochastic diagnostic improved mean Dice from `0.10549` to
  `0.22692` but worsened endpoint error from `118.60` to `149.95 mm`; all six
  episodes still missed the endpoint. Pt14 averaged roughly `0.3730` Dice and
  `122.76 mm`, while pt18 averaged `0.0809` Dice and `177.13 mm`. This is
  meaningful coverage learning, but it fails the preregistered endpoint and
  joint-improvement gates. Do not extend this reward unchanged to 1M.
- Audited one endpoint-strengthening candidate by changing only GDT scale
  `0.1 -> 1.0` while retaining coverage `500`. Exact full covering routes
  remain much more valuable than endpoint shortcuts:
  - pt14 full/direct return `+310.83/+30.54`;
  - pt18 full/direct return `+301.84/+31.22`;
  - both full routes retain Dice above `0.52` and complete traversal;
  - novel oracle actions are positive in `86.67%` of sampled states;
  - all sampled reward maxima stay on-target, while `98.44%` are
    endpoint-progressing.
  Thus stronger GDT does not make the low-Dice direct path optimal, does not
  reward off-target shortcuts, and preserves bounded cumulative coverage.
- Register
  `scripts/run_navigator_bomopi_compact_cov500_gdt1_512k.sh`: resume the exact
  256k state, retain entropy `0.0001` and the frozen `5e-6` learning rate, and
  change only GDT `0.1 -> 1.0` through 512,000 total frames. Validate near
  307k, 410k, and the mandatory final state. Acceptance requires at least one
  held-out endpoint reach, deterministic mean Dice at least `0.10` with
  endpoint error below `75 mm`, and a final fixed-seed stochastic mean Dice
  at least `0.20` with endpoint error below `120 mm`. If no endpoint is reached
  by 512k, stop this pure PPO line rather than run it to 1M.
- `navigator-bomopi-gru-compact-cov500-gdt1-512k-v1.service` resumed the exact
  256,000-frame / 2,500-update state and completed successfully in 22m03s.
  The effective configuration retained coverage `500`, entropy `0.0001`,
  learning rate `5e-6`, image-only policy observations, and changed only GDT
  `0.1 -> 1.0`. Invalid actions remained exactly zero; maximum observed KL was
  `0.02133`, all registered PPO epochs completed, and stochastic training
  diversity remained high (final-ten mean `0.98918`). Peak CUDA usage was
  `9,391.9 MiB` allocated and `13,414 MiB` reserved. A non-fatal 20 MiB
  expandable-segment mapping warning occurred during final validation, as in
  the preceding 256k run, but both cases and the final checkpoint completed.
- Deterministic held-out gates were:
  - 307,200: Dice `0.053576`, endpoint `128.84 mm`, diversity `0.05664`,
    boundary residence `0.00098`;
  - 410,112: Dice `0.072309`, endpoint `122.13 mm`, diversity `0.22070`,
    boundary residence `0.00049`;
  - 512,000: Dice `0.036879`, endpoint `169.78 mm`, diversity `0.05469`,
    boundary residence `0.11914`.
  Every case executed all actions, but no gate reached an endpoint or completed
  a traversal. The final pt14 result was Dice `0.07163` / endpoint
  `92.28 mm`; pt18 regressed to Dice `0.00213` / endpoint `247.28 mm` with
  `0.2373` boundary residence. This fails every preregistered deterministic
  gate and rules out extending this exact reward continuation to 1M.
- The final-ten stochastic training means were total reward `+0.01617`,
  GDT reward `+0.00663`, coverage reward `+0.04050`, target-distance reward
  `-0.02007`, and value loss `0.41991`. Maximum action probability increased
  from roughly `0.03` early in the continuation to a final-ten mean `0.06021`
  (maximum `0.07917`), but the stronger mode was not a coherent endpoint route.
- The fixed final stochastic diagnostic, using seeds `101`, `202`, and `303`
  on exactly pt14/pt18, produced:
  - mean Dice `0.255286`;
  - mean endpoint distance `138.390 mm`;
  - diversity `0.99349`, boundary residence `0.02962`;
  - zero endpoint reaches and zero traversals across all six episodes.
  Pt14 Dice was `0.45609/0.48094/0.43615` with endpoint distances
  `87.39/116.06/99.18 mm`; pt18 Dice was `0.06794/0.02533/0.06527` with
  endpoint distances `151.22/218.12/158.35 mm`. Compared with the 256k
  checkpoint, expected Dice improved (`0.22692 -> 0.25529`) and endpoint error
  improved modestly (`149.95 -> 138.39 mm`), but the preregistered
  `<120 mm` endpoint gate and mandatory endpoint reach both fail. The dominant
  remaining failure is case generalization plus diffuse/non-coherent action
  selection, not invalid movement or a universal inability to cover bowel.
- The first standalone stochastic-evaluation attempt imported Navigator from
  the shared uv environment's editable checkout rather than this execution
  worktree. It failed before any rollout because the stale `Config` signature
  rejected current fields. `scripts/evaluate_navigator_stochastic.py` now
  prepends its adjacent repository `src` directory before importing Navigator.
  A remote `uv` import/CLI smoke passed, and the six reported episodes were
  generated only after that fix. Training was unaffected because the managed
  BOMOPI launcher already exports the execution worktree's `src` through
  `PYTHONPATH`.

### M27: Supervised perception upper bound for the repaired control stack

- Stop the failed image-only pure-PPO continuation at 512k as preregistered.
  The next causal question is whether the remaining failure comes primarily
  from the weak image-derived state or from recurrent control/optimization.
- Register
  `scripts/run_navigator_bomopi_gtmask_compact_cov500_gdt1_256k.sh` as an
  explicitly supervised upper bound:
  - train from scratch; do not reuse incompatible three-channel weights;
  - retain the 156-action bounds-masked direction/length likelihood, exact
    integer movement, GRU, 60-mm patch, 9-mm action, 6-mm path radius,
    2,048-step horizon, coverage `500`, GDT `1`, recovery `0.2`, graded
    target-distance cost, and pt14/pt18 held-out split;
  - change the policy state to the existing seven-channel supervised contract:
    CT, four image-filter responses, cumulative path, and local GT mask;
  - use entropy `0.0001`; anneal to the `5e-6` floor by 65,536 frames; validate
    only near 102.4k, 204.8k, and the mandatory 256k final state.
- This run uses GT segmentation at policy inference and can never be reported
  as annotation-free, image-only, or clinically deployable. Its purpose is to
  locate the bottleneck. First require a 4,096-frame CUDA smoke with finite
  PPO, zero invalid actions, exact seven-channel startup, and no OOM.
- Promotion beyond 256k requires at least one endpoint reach, or mean Dice at
  least `0.30` together with endpoint distance below `50 mm`; also reject a
  boundary/short-cycle mode even if accumulated Dice is high. If the
  supervised upper bound fails, stop tuning policy observations and pivot the
  main method toward explicit route planning/energy minimization. If it passes,
  use it only to justify replacing the GT mask with a learned/self-supervised
  image representation.
- The first 4,096-frame smoke verified the exact seven-channel configuration,
  fresh initialization, and intended reward/action settings. Training completed
  all 4,096 frames, but validation OOMed on pt18 while allocating a 100-MiB
  boundary-padded filter volume. PyTorch still held 14.10 GiB from training and
  the preceding 2,048-step pt14 rollout; this is a validation memory-lifetime
  defect, not evidence against the supervised policy.
- Implement two accuracy-neutral memory repairs before retrying:
  - `get_patch` now allocates only the requested fixed-size output and copies
    the in-volume intersection when a patch crosses a boundary, exactly
    equivalent to constant-padding the complete volume and then slicing;
  - validation moves the retained path-mask report artifact to CPU and deletes
    each full 2,048-step GPU rollout before loading the next subject.
  Exact interior, low-boundary, high-boundary, multichannel, and supervised
  channel regressions pass. The complete remote Navigator suite passes under
  `uv`: `104 passed`, four subtests passed, 18 warnings.
- The fresh `smoke4k-v2` rerun completed all 4,096 training frames and both
  held-out 2,048-step validations in 46.5s. Training peak CUDA telemetry was
  `10,645.7/12,014 MiB` allocated/reserved; complete-process validation peak
  was `14,590/14,828 MiB`. One 20-MiB expandable-segment mapping warning
  remained during validation, but no allocation failed and the final
  checkpoint/metrics were saved.
- Technical promotion gate passes: seven policy channels, no reload, all five
  PPO epochs, KL `0.00464`, value loss `0.4087`, exactly zero invalid actions,
  and stochastic training diversity `0.9891`. Untrained deterministic smoke
  quality is Dice `0.03672`, endpoint error `192.59 mm`, diversity `0.03516`,
  boundary residence `0.3840`, and zero endpoint/traversal success; this is not
  treated as learned evidence.
- The 256k supervised run reached all training frames in 22m40s. In-process
  deterministic gates were:
  - 102,400: Dice `0.041010`, endpoint `81.77 mm`, positive GDT `0.28784`,
    diversity `0.04883`;
  - 204,800: Dice `0.185604`, endpoint `49.55 mm`, positive GDT `0.19653`,
    diversity `0.02930`.
  Both cases were balanced at 204.8k (Dice `0.19470/0.17650`, endpoint
  `57.18/41.92 mm`) with positive returns, but neither reached the endpoint.
- The periodic `checkpoint_256000.pth` was saved before mandatory final
  validation. That in-process validation OOMed while TensorDict attempted to
  stack 2,048 seven-channel observations into a 3.42-GiB tensor with
  10.83 GiB of training allocations still live. This does not invalidate the
  checkpoint. Add a fresh-process deterministic mode to
  `scripts/evaluate_navigator_stochastic.py`; normalize string/device inputs in
  validation; the complete Navigator suite remains `104 passed`, four subtests
  passed. Fresh-process final deterministic scoring gives Dice `0.116015`,
  endpoint `83.66 mm`, diversity `0.04297`, and zero traversal, so deterministic
  ranking retains the 204.8k state.
- Fixed three-seed stochastic scoring changes the conclusion:
  - 204.8k: mean Dice `0.457375`, endpoint `50.05 mm`, zero endpoint reaches
    and zero traversals in six episodes;
  - 256k: mean Dice `0.420629`, endpoint `30.58 mm`, endpoint/traversal success
    `2/6 = 0.3333`, diversity `0.98372`, and boundary residence `0.00033`.
  At 256k, pt18 completes in seeds 101 and 202 with Dice `0.41604/0.43696`,
  endpoint error `2.60/1.50 mm`, and 1,820/1,922 actions. Pt14 obtains Dice
  `0.37379-0.48937` but no endpoint reach. These are genuine joint successes:
  both successful episodes exceed Dice `0.40` and satisfy the independent
  3-mm endpoint test; they are not reward-only or shortcut successes.
- Training remained stable: maximum KL `0.01836`, invalid actions exactly zero,
  final-ten diversity `0.99168`, final-ten coverage reward `+0.07013`, and
  final-ten maximum action probability `0.06256`. The GT mask therefore fixes
  the severe pt18 perception/generalization failure and proves the compact
  policy can learn a successful action distribution. Deterministic argmax
  remains a poor decoder of that distribution.
- Register one unchanged continuation:
  `scripts/run_navigator_bomopi_gtmask_compact_cov500_gdt1_512k.sh`. Resume
  the exact 256k policy, critic, optimizer, scheduler, and frame/update counters;
  keep the `5e-6` floor and every environment/reward/policy setting unchanged.
  At 512k, repeat the same six stochastic episodes. Promotion requires mean
  Dice at least `0.40`, traversal success at least `0.50`, and at least one
  pt14 traversal; otherwise stop this supervised PPO continuation and address
  decoding/planning rather than add more frames.
- The exact continuation restored 256,000 frames / 2,500 updates and completed
  512,000 total frames in 23m06s. Peak CUDA allocation/reservation was
  `14,639.8/14,806 MiB`; all actions executed and the final model plus all
  registered gates were saved.
- Deterministic gates remained incomplete:
  - 307,200: Dice `0.281099`, endpoint `57.38 mm`, diversity `0.17773`;
  - 410,624: Dice `0.228237`, endpoint `49.87 mm`, diversity `0.04297`;
  - 512,000: Dice `0.234988`, endpoint `100.85 mm`, diversity `0.05469`.
  No deterministic episode reached an endpoint or traversed. The pt14/pt18
  asymmetry changed between gates rather than converging monotonically.
- Fixed three-seed stochastic results also regress after the 256k champion:
  - 256k: Dice `0.420629`, endpoint `30.58 mm`, traversal `2/6`;
  - 410,624: Dice `0.392909`, endpoint `35.49 mm`, traversal `0/6`;
  - 512k: Dice `0.405908`, endpoint `41.98 mm`, traversal `0/6`.
  At 410,624 and 512k neither pt14 nor pt18 reaches the endpoint in any fixed
  rollout. Thus the continuation fails both the `>=0.50` traversal-rate and
  pt14-success gates. Retain the 256k checkpoint as the traversal-first
  supervised champion and stop this PPO line.
- Continued PPO sharply concentrated the categorical policy without improving
  completion. Maximum action probability rose from a final-ten mean `0.06256`
  at 256k to `0.31261` at 512k (maximum `0.36608`), while final-ten value loss
  rose to `1.2911`. Training still looked superficially healthy: final-ten
  diversity `0.98322`, reversals `0.00587`, coverage reward `+0.08080`, GDT
  reward `+0.01313`, maximum KL `0.02578`, and zero invalid actions. This is
  over-specialization of a locally profitable stochastic policy, not an
  invalid-action, boundary, or two-point-cycle exploit.
- Scientific conclusion from the upper bound:
  - local GT anatomy removes the image-only pt18 failure and permits real
    joint Dice/endpoint traversal, proving movement, likelihood, and the
    2,048-step task are executable;
  - pure PPO does not make that success reliable or transfer it to pt14;
  - deterministic categorical mode is not a valid proxy for expected
    performance, but sampling alone is also not a reliable deployable decoder;
  - further work should change closed-loop decoding/planning or introduce
    explicit route supervision, while the primary annotation-free path must
    replace the privileged GT channel rather than conceal it.

### M28: Cross-patient image-filter signal audit

- Before spending more frames on image-only PPO, audit whether its five
  image-derived channels contain a stable bowel/background distinction across
  the two held-out patients. `scripts/audit_navigator_filter_signal.py` samples
  up to 100,000 bowel voxels and an equal number of non-bowel voxels within
  30 mm of bowel for pt14 and pt18. It reports per-channel ROC AUC and fits a
  five-channel standardized logistic model on one patient before evaluating it
  unchanged on the other.
- Ground-truth segmentation is used only to label this offline diagnostic. It
  is never passed to the policy, used to train a replacement representation,
  or used to tune an environment observation. The audit therefore measures
  information and transfer; it does not produce an annotation-free model.
- Direction-free single-channel AUCs show substantial within-patient signal:

  | channel | pt14 | pt18 |
  | --- | ---: | ---: |
  | clipped CT | 0.865846 | 0.620469 |
  | dark tubularity | 0.702547 | 0.707311 |
  | bright tubularity | 0.786327 | 0.536060 |
  | band pass | 0.700759 | 0.627068 |
  | gradient | 0.730097 | 0.582987 |

  The dark-tubularity response is mostly zero inside bowel
  (`68.1%/59.0%` for pt14/pt18) but less often zero in the surrounding shell
  (`24.7%/29.7%`), so its useful direction is consistently negative. In
  contrast, the direct polarity of bright tubularity, band pass, and gradient
  changes or nearly disappears on pt18.
- Cross-patient transfer is weak despite balancing the classes and exposing all
  five channels: training on pt14 and testing on pt18 gives AUC `0.530584`;
  training on pt18 and testing on pt14 gives `0.593630`; mean cross-patient AUC
  is only `0.562107`. This is barely above random ranking and is consistent
  with the observed image-only pt18 failure.
- Conclusion: movement and the compact categorical policy are no longer the
  primary blocker. The current handcrafted filter bank has patient-specific
  signal but does not provide a stable cross-patient bowel representation.
  More PPO on the same observations is not justified by this evidence. The
  next bounded experiment must alter the annotation-free representation or
  use an explicit image-derived energy/planner; it must not use the supervised
  upper bound's GT channel or choose transforms from held-out labels.
- The bidirectional two-case audit is intentionally pessimistic because the
  actual policy learned from 15 patients, not one. Pool the exact seeded
  training split (25,000 bowel and 25,000 local-shell samples per patient) and
  fit once before opening pt14/pt18. The pooled model has training AUC
  `0.854088`, test AUC `0.897297` on pt14, and only `0.555567` on pt18. Thus
  the features transfer very well to pt14 but fail specifically on pt18,
  matching the image-only PPO result rather than indicating a universal lack
  of image signal.
- Training-case-only direction checks are perfectly consistent across all 15
  subjects: direct CT, bright-tubularity, band-pass, and gradient AUC are above
  `0.5` in every subject, while dark-tubularity AUC is below `0.5` in every
  subject. Held-out subset diagnostics show the exceptional pt18 shift:
  CT-only AUC is `0.613767`, inverse dark-only AUC is `0.705065`, but
  band-pass and gradient reverse to `0.377009/0.416953`. On pt14 the same
  channels score `0.864568/0.702031/0.699363/0.723592`.
- This audit also exposes an unrun causal control. The failed image-only run
  used the three-channel `shin_068_repaired` state (CT, dark response, thin
  path), while the successful supervised run simultaneously changed to seven
  channels (CT, all four filters, dilated path, GT mask). The gain therefore
  cannot yet be attributed only to GT. Register
  `scripts/run_navigator_bomopi_filters_compact_cov500_gdt1_256k.sh` as the
  exact six-channel no-GT control: same fresh seed, compact actions, GRU,
  reward, schedule, and validation; remove only the segmentation channel from
  the supervised upper bound. Require a 4,096-frame CUDA smoke before the
  256k run. Score the same six fixed stochastic episodes at 256k; any real
  traversal would overturn the stronger claim that privileged perception was
  necessary, while failure preserves the pt18 representation diagnosis.
- The six-channel 4,096-frame CUDA smoke completed successfully with the exact
  15/2 split, `observe_segmentation=false`, no checkpoint reload, and
  `observation_channels=6`. All eight updates completed five PPO epochs;
  invalid-action fraction was exactly zero, maximum KL was `0.006150`, final
  value loss was `0.37817`, and stochastic recent-position diversity was
  `0.99074`. Peak CUDA allocation/reservation was
  `12,615.1/12,664 MiB`, safely below the 15.5-GiB device limit.
- Untrained deterministic validation is not interpreted as efficacy: final
  Dice was `0.011563`, endpoint error `249.41 mm`, diversity `0.09961`,
  boundary residence `0.63062`, and no endpoint/traversal succeeded. The smoke
  passes only the technical promotion gate. Launch the fresh 256k control with
  the preregistered sparse validation interval.
- The fresh six-channel control completed all 256,000 frames in 22m46s and
  saved `checkpoint_256000.pth`. All 500 updates completed five PPO epochs;
  invalid-action fraction stayed exactly zero, maximum KL was `0.014195`,
  final-ten value loss was `0.39838`, final-ten stochastic diversity was
  `0.99012`, and final-ten maximum action probability was only `0.01936`.
  Training peak allocation/reservation was `12,654.7/12,922 MiB`; final
  validation raised reservation to `14,496 MiB` and emitted one non-fatal
  20-MiB expandable-segment mapping warning, but the service exited cleanly.
- Deterministic validation did not learn a route:
  - 102,400: Dice `0.015947`, endpoint `139.95 mm`, diversity `0.03125`,
    boundary residence `0.00952`;
  - 204,800: Dice `0.002517`, endpoint `142.36 mm`, diversity `0.02734`,
    boundary residence `0.35669`;
  - 256,000: Dice `0.006887`, endpoint `125.17 mm`, diversity `0.24219`,
    boundary residence `0.00098`.
  No endpoint or traversal succeeded at any gate. A fresh-process mode
  evaluation exactly reproduces the final in-process metrics.
- Fixed stochastic seeds 101/202/303 give mean Dice `0.236538`, endpoint
  distance `104.23 mm`, diversity `0.98893`, boundary residence `0.07096`,
  and zero endpoint/traversal successes in six episodes. Pt14 Dice is
  `0.41383/0.42939/0.28847` with endpoint errors
  `55.76/86.30/54.68 mm`; pt18 Dice is only
  `0.14222/0.00506/0.14026` with endpoint errors
  `145.32/182.40/100.91 mm`.
- Causal conclusion: adding the three extra handcrafted filters improves the
  failed three-channel checkpoint's stochastic endpoint error
  (`149.95 -> 104.23 mm`) only modestly and produces no traversal. It cannot
  explain the supervised checkpoint's two genuine pt18 traversals and
  `0.42063` mean Dice. The local GT channel is therefore the component that
  caused the upper-bound success. Do not continue this six-channel PPO policy
  to 512k. The next representation experiment must address the pt18 domain
  shift explicitly rather than add frames or reward terms.

### M29: Dark-tubularity domain-robustness ablation

- The 15-case audit shows that dark-tubularity is the only filter whose useful
  direction remains unchanged on the exceptional pt18 validation subject:
  inverse AUC is `0.70506` on pt18 and `0.70203` on pt14. The six-channel
  policy can ignore this weaker invariant cue in favor of easier CT/gradient
  correlations that reverse on pt18.
- Add the explicit `navigation_dark_path` policy-observation contract. It
  exposes only the raw versioned dark-tubularity patch and the agent-owned
  dilated cumulative path patch. It retains label-free time, absolute
  position, and previous-direction context, but removes CT, bright
  tubularity, band-pass, and gradient. GT segmentation input is rejected by
  configuration validation. Dataset loading still computes the same cached
  image-only filter bank; no target label is used to construct either policy
  channel.
- This ablation is selected using pt18 validation behavior and therefore
  diagnoses domain robustness rather than providing an unbiased final test.
  Any eventual research claim requires a frozen, untouched patient test set.
- Register
  `scripts/run_navigator_bomopi_darkpath_compact_cov500_gdt1_256k.sh` with the
  exact control seed, 15/2 split, compact 156 actions, GRU, reward, 60-mm
  patch, 2,048-step horizon, entropy, and learning-rate schedule. The only
  model change is `6 -> 2` observation channels. The full remote `uv` suite
  passes: `105 passed`, four subtests passed, 18 warnings.
- Require a fresh 4,096-frame CUDA smoke with exactly two channels, finite
  five-epoch PPO, zero invalid actions, and complete two-case validation.
  Then run fresh to 256k and score seeds 101/202/303. Promotion beyond 256k
  requires at least one held-out traversal or, at minimum, pt18 mean Dice
  `>=0.20` with pt18 endpoint error `<=80 mm` and overall mean Dice above the
  six-channel control's `0.23654`. Otherwise reject the forced invariant cue
  and do not add more PPO frames.
- The fresh 4,096-frame CUDA smoke confirms
  `policy_observation_contract=navigation_dark_path`,
  `observation_channels=2`, the exact seeded split, no reload, and no GT
  observation. All eight updates completed all five epochs; maximum KL was
  `0.007519`, final value loss `0.51055`, invalid actions exactly zero, and
  stochastic diversity `0.99271`. Training peak allocation/reservation was
  only `7,014.4/7,858 MiB`; complete-process reservation reached
  `10,900 MiB`, leaving substantially more memory headroom than the
  six/seven-channel states.
- Untrained mode validation is Dice `0.006102`, endpoint `216.24 mm`,
  diversity `0.09961`, boundary residence `0.93262`, and zero success. This is
  not efficacy evidence, but the technical smoke gate passes. Promote the
  exact fresh 256k ablation.
- The fresh two-channel run completed 256,000 frames in 20m53s. All 500
  updates completed five PPO epochs; invalid actions remained exactly zero,
  maximum KL was `0.011661`, final-ten value loss `0.27463`, stochastic
  diversity `0.99115`, and maximum action probability only `0.02707`.
  Training peak allocation/reservation was `7,014.4/9,344 MiB`; complete
  validation raised reservation to `11,208 MiB`.
- Deterministic mode fails at every gate:
  - 102,400: Dice `0.003190`, endpoint `172.59 mm`, boundary `0.99365`;
  - 204,800: Dice `0.011268`, endpoint `211.08 mm`, boundary `0.33911`;
  - 256,000: Dice `0.003200`, endpoint `216.98 mm`, boundary `0.97705`.
  All gates have zero endpoint/traversal success. The policy's categorical
  mode repeatedly becomes a boundary direction even though stochastic
  training remains diverse.
- Fixed stochastic seeds 101/202/303 also reject the hypothesis: mean Dice
  `0.122273`, endpoint `188.60 mm`, diversity `0.99284`, boundary residence
  `0.24601`, and zero endpoint/traversal successes. Pt14 Dice is
  `0.15996-0.19390`; pt18 Dice is `0.03060-0.13227` with endpoint errors
  `227.10-243.29 mm`. This misses every preregistered promotion condition and
  is substantially worse than the six-channel control.
- Conclusion: consistent voxel-level bowel/background ranking is insufficient
  for directional path control. Forcing PPO to use dark-tubularity discards
  spatial/intensity cues without solving the pt18 shift. Stop this line at
  256k. The next method must learn or construct a spatial bowel-likelihood
  field/energy and plan through it; it should not be another subset of the
  same four filters or another PPO reward-scale ablation.

### M30: Seed-conditioned image-likelihood audit

- Test a label-free, patient-adaptive alternative before implementing an
  energy planner. `scripts/audit_navigator_seed_likelihood.py` constructs a
  feature prototype from the supplied start seed using clipped CT and the
  cached image-only filter bank. Per-channel deviations are normalized by
  subject-specific 10th-90th percentile ranges. It then reports how well
  negative prototype distance ranks bowel above a 30-mm local background
  shell.
- GT segmentation is used only to sample/report ROC AUC and to verify the
  audit seed. It is not used in the prototype or score. The BOMOPI cached
  start itself was derived from annotations; a deployable tracker would
  require the equivalent externally supplied start seed already assumed by
  the annotation-free loader.
- A nominal 3-mm seed ball is not safe on these cached starts: only `45.5%`
  of its 33 voxels are target on pt14 and `54.5%` on pt18. The resulting
  CT+dark score is strong on those validation cases (`0.768/0.830` AUC) but
  has training mean/minimum AUC only `0.597/0.132`. This is a contaminated
  prototype, not a reliable bowel-likelihood field.
- Restricting the prototype to the exact guaranteed seed voxel removes that
  contamination and yields CT+dark AUC `0.80011/0.81586` on pt14/pt18.
  However, the 15-case training mean/minimum remain only `0.61825/0.14146`.
  It reverses on pt11, pt6, and pt2 (`0.1415/0.2554/0.2951`) while reaching
  `0.79-0.86` on several others. Dark-only exact-seed similarity is more
  stable but weak (training mean/minimum `0.57776/0.32577`; held-out
  `0.70151/0.70654`).
- Conclusion: subject conditioning can repair pt18's global distribution
  shift, but one static seed appearance does not represent intensity/content
  changes along the entire bowel. Do not expose this scalar as a replacement
  GT channel or plan globally from it. A viable annotation-free energy must
  include spatial tube orientation/continuity and probably conservative
  online appearance adaptation; its audit should measure directional
  alignment and connected traversal, not only voxel ROC AUC.

### M31: Label-free Hessian orientation and energy tracking

- Test whether clipped CT contains a usable local direction field before
  spending more PPO frames. `scripts/audit_navigator_hessian_orientation.py`
  compares the minimum-absolute-eigenvalue Hessian axis against local PCA
  tangents of the audit-only bowel skeleton. Planning inputs never include the
  skeleton, segmentation, endpoint, GDT, or Dice.
- A straight synthetic tube gives exactly zero median angular error. Across all
  17 BOMOPI cases, the 6-mm Hessian has mean absolute cosine `0.69759`, median
  angle `33.48 deg`, `45.96%` within 30 degrees, and `60.06%` within 45
  degrees, versus the unoriented random-axis baselines `0.5`, `60 deg`,
  `13.40%`, and `29.29%`. The top half by axis-gap confidence improves to
  cosine `0.74855`, median `27.76 deg`, and `52.48%` within 30 degrees.
  Three and 9 mm are weaker (`42.94/37.86 deg` median). Held-out pt14/pt18
  retain real but imperfect 6-mm signal (`35.92/38.88 deg` median).
  Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-orientation-train15-test2.json`.
- `scripts/evaluate_navigator_hessian_streamline.py` integrates the continuous
  6-mm axis in floating-point voxel coordinates, aligns its sign to the
  previous move, uses midpoint integration, plans both initial orientations,
  and selects without labels. It fails on pt14/pt18: mean Dice `0.03045`,
  endpoint `105.13 mm`, and zero traversal. Step-size/momentum diagnostics
  (`1.5/3.0 mm`, momentum `0.75/0.9`) do not repair it. The field follows
  other tubular anatomy, so sub-voxel integration is not the remaining
  blocker. Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-streamline-6mm-step4.5-v1.json`.
- Implement `scripts/evaluate_navigator_hessian_energy_beam.py` as a
  label-free continuous beam tracker. It uses clipped CT Hessians, the cached
  image-only dark response, one supplied start seed, 26 physical unit
  directions, continuous positions, trilinear sampling, a 60-degree turn
  limit, and post-planning audit metrics. Target and endpoint files are opened
  only after planning. The two unoriented seed-axis branches are searched
  independently, then selected by image energy.
- The first synthetic test exposed two invalid objective details:
  - a global visited set let one beam branch penalize another. Revisit memory
    is now path-local, full-episode by default, and contains every undilated
    voxel rasterized along each 4.5-mm segment;
  - all-positive terms forced paths to grow forever. A stop action is now the
    best-scoring prefix under an explicit per-step cost. On a finite bright
    synthetic tube, the selected branch follows `[4,16,16]` to `[34,16,16]`
    in ten moves and stops rather than turning into background. This behavior
    is covered by `tests/test_navigator_hessian_energy_beam.py`.
- Reject two reward-hacking variants before real validation:
  - `dark_weight * (1-dark)` is nearly a constant positive reward because the
    Meijering cache is black over most of the volume. V5 drives its selected
    mean dark penalty to only `0.03-0.05` while following wrong anatomy and
    obtains mean training Dice only `0.05472`;
  - direct positive CT reward leaves a synthetic bright background instead of
    following a dark tube. Static seed-CT similarity is also disabled because
    M30 shows it reverses across subjects and pt11's later on-bowel CT differs
    strongly from its seed. Default CT and seed-CT weights are therefore zero.
- Ratios alone also reward numerical structure in nearly flat regions. Add a
  deterministic subject-specific 95th-percentile scale for the middle
  absolute Hessian eigenvalue, estimated from up to 200,000 image voxels.
  Multiply axis and transverse-balance evidence by clipped normalized
  strength, add `0.25 * strength`, retain `-2 * dark`, `-1 * revisit`, and a
  `-1.75` step cost. On the three calibration cases pt1/pt11/pt6 with beam 32:
  - 512 moves: mean Dice `0.20192`, endpoint `125.81 mm`
    (`0.22631/0.24148/0.13797`);
  - 1,024 moves: mean Dice `0.24791`, endpoint `104.28 mm`
    (`0.31037/0.24197/0.19140`);
  - 2,048 moves: mean Dice `0.29583`, endpoint `63.96 mm`
    (`0.39167/0.23280/0.26301`).
  No case reaches the strict 3-mm endpoint or traversal gate. Prefix Dice
  generally improves with horizon for pt1/pt6, while pt11 peaks at `0.26636`
  near 768 moves and then accumulates false positives.
- Audit every proposed discriminator numerically on training paths:
  - Hessian bright polarity separates selected on/off-bowel points strongly
    for pt1 (`78.0/24.4%` both-negative) and pt6 (`95.3/23.7%`), less so for
    pt11 (`93.6/70.8%`). Hard gating improves pt1 at 512 moves
    (`0.226 -> 0.354`) but makes pt6 collapse (`0.138 -> 0.0028`); a soft
    centered polarity term also fails to improve mean Dice. Polarity remains
    an optional ablation and defaults to unsigned/zero weight.
  - Fixed 3-mm Hessians improve pt1/pt11 to `0.29667/0.28172` but collapse pt6
    to `0.00507`; 9 mm gives mean `0.17559`; 6 mm remains the safest mean.
    Image score cannot select a scale because the catastrophic pt6 3-mm path
    has the highest score. Multiscale pairwise axis consensus is also
    non-discriminative on/off bowel, and minimum strength reverses on pt6.
  - Concatenating both 2,048-step seed branches increases false-positive
    volume and lowers mean Dice from `0.29583` to `0.27241`; retain
    image-energy branch selection.
- Increasing beam width from 32 to 128 is the first search-only improvement.
  At 512 moves it raises mean Dice `0.20192 -> 0.22237` and reduces final
  endpoint error `125.81 -> 49.16 mm`. At 1,024 moves it reaches mean Dice
  `0.32827`, final endpoint `84.44 mm`, and mean minimum-over-path endpoint
  `13.79 mm`: pt1 `0.39867`, pt11 `0.32755`, pt6 `0.25860`. Pt11 passes
  within `4.44 mm` of the endpoint but does not meet the 3-mm gate. Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-energy-beam-train3-width128-1024-v14.json`.
- The width-128, 2,048-step promotion
  (`navigator-energy-width128-2048-v15.service`) completes successfully, but
  does not justify the longer horizon. Mean Dice is `0.33004`, only `0.00176`
  above 1,024, while pt1/pt11 regress to `0.35604/0.28503`, pt6 improves to
  `0.34904`, and final endpoint error worsens to `148.42 mm`. Energy-prefix
  means decline in all three cases and cannot distinguish the useful pt6
  extension from the harmful pt1/pt11 extensions. Freeze width 128 and 1,024
  steps as the parsimonious held-out configuration; do not select a per-case
  prefix using GT. Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-energy-beam-train3-width128-2048-v15.json`.
- Launch exactly one frozen evaluation on pt14/pt18 as
  `navigator-energy-heldout-width128-1024-v16.service`. No further objective,
  scale, horizon, or beam-width tuning may use these two cases.
- The frozen service exits successfully, but held-out performance rejects the
  handcrafted energy as a deployable solution: mean Dice is `0.10061`, final
  endpoint error `95.75 mm`, minimum-over-path endpoint error `23.87 mm`, and
  there are zero endpoint hits or traversals. Pt14 reaches Dice `0.20038` with
  final/minimum endpoint `57.78/19.75 mm`; pt18 reaches only `0.000852` with
  `133.71/27.99 mm`.
- This is not a branch-selection technicality that can be repaired after
  seeing held-out labels. On pt18, image energy strongly selects the negative
  branch (score `495.11`, Dice `0.000852`) over the alternative (score
  `365.61`, Dice `0.06019`). Both fail, and selecting the latter from Dice
  would be an oracle. Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-energy-beam-heldout2-width128-1024-v16.json`.
- Conclusion: continuous movement, segment-rasterized full-path revisit,
  bidirectional search, robust Hessian magnitude, and a fourfold wider beam
  materially improve the three development subjects, proving that the
  original movement/search defects were real. They do not solve the
  cross-patient representation problem already exposed by M28-M30. Do not
  tune this energy on pt14/pt18 or claim the `>=0.40`/full-traversal goal.
  Preserve it as a reproducible negative baseline and possible proposal
  generator. The next research stage must learn a patient-robust
  annotation-free representation (or explicitly relax the no-annotation
  constraint for training); more handcrafted weights, horizon, or PPO frames
  over the same channels are not supported by these results.

### M32: Start-seed audit and label-free image pretraining

- Diagnose whether the frozen Hessian tracker's held-out failure is merely a
  bad start position. The cached start is generated by mapping an anatomical
  landmark to the nearest small-bowel-mask voxel, so it often lies on the
  target boundary rather than the lumen center. Pt1, pt11, pt14, and pt18
  have only `1.5 mm` target clearance at the cached start; their distances to
  the audit-only target skeleton are `10.92/10.06/12.19/6.71 mm`
  respectively. Pt6 is not enough to make this mechanism reliable.
- A CT-only transverse Newton recentering rule is not a safe preprocessing
  fix. It improves pt18's audit-only skeleton distance from `6.71` to
  `2.47 mm`, but exits the target on pt1, pt11, and pt14. Keep it rejected
  rather than selecting per-patient recentering behavior from labels.
- Add `--start-voxel X Y Z` to the Hessian evaluator for an explicit,
  single-case causal audit. This override does not infer or expose a start
  from GT, and the artifact records `start_source`; using an audit-derived
  coordinate is nevertheless an oracle experiment, not a deployable result.
  Starting pt18 at its nearest skeleton voxel `[122,45,163]` still gives Dice
  only `0.001665` for the image-energy-selected branch (`0.02213` for the
  alternative), with no endpoint hit or traversal. Artifact:
  `/home/matey/project/segmentor/results/navigator_bomopi/hessian-energy-beam-audit-oracle-center-pt18-v17.json`.
  Therefore seed centering is not sufficient to repair the representation.
- Implement a label-free 3-D masked denoising pretraining stage for the shared
  Navigator encoder. Inputs are clipped CT plus the four cached image filters;
  the script never opens a segmentation, endpoint, GDT, or Dice file. Cubical
  masks and Gaussian noise corrupt random body patches, and Smooth-L1 is
  measured only over hidden voxels. The first smoke exposed and removed an
  invalid all-voxel baseline that was dominated by unchanged copied voxels.
- Pretrained convolutional weights load into recurrent PPO with strict key and
  shape validation. A five-channel image checkpoint expands to the six-channel
  image-plus-visitation policy by copying the five image kernels exactly and
  zero-initializing the visitation kernel; the policy-specific context
  projection remains independently initialized. Checkpoints use
  `torch.load(..., weights_only=True)`.
- A 200-step CUDA throughput run with batch 64 uses `1,408/1,792 MiB`
  allocated/reserved and sustains `17.9` updates/s. On fixed unlabeled
  pt1/pt11 validation patches, masked reconstruction falls to `0.009715`
  versus the honest corrupted mean-fill baseline `0.018518`. This establishes
  that the task is learned rather than copied, but is not yet evidence of
  navigation or bowel specificity.
- Launch the exact 20,000-step, 12-training-case run with pt1/pt11/pt6 used
  only for image-reconstruction validation as
  `navigator-image-pretrain-bomopi-v1.service`. Pt14/pt18 remain sealed after
  the frozen M31 result. The best checkpoint is atomically replaced after
  each validation, TensorBoard logs are written under the matching run name,
  and `scripts/run_navigator_bomopi_image_pretrain.sh` records the complete
  `uv` invocation.
- The 20,000-step service completes successfully in `1,074.6 s` at
  `18.61` updates/s. Best validation is `0.002199` at step 17,750 versus the
  masked mean-fill baseline `0.020219`, a `9.19x` reduction. Peak CUDA
  allocation/reservation is only `1,408.4/1,792 MiB`. The atomically saved
  checkpoint records the exact 12/3 split and `label_files_read=false`.
- Add generic immutable split manifests to the legacy BOMOPI loader, then
  create a 15-case development view that excludes frozen pt14/pt18. Both
  matched PPO arms use seed 42, training cases
  pt2/pt4/pt5/pt7/pt9/pt12/pt15/pt17/pt19/pt20/pt21/pt23, and validation
  cases pt1/pt11/pt6. The only intended difference is encoder initialization.
  A 4,096-frame pretrained smoke passes checkpoint expansion, PPO, all three
  validation cases, and artifact saving before either matched 64k run starts.
- Deterministic-mode development results at 16,384/32,768/49,152/65,536
  frames are:
  - random initialization Dice
    `0.008325/0.008166/0.042084/0.006086`, endpoint error
    `251.94/165.64/215.06/203.51 mm`;
  - pretrained initialization Dice
    `0.012563/0.010060/0.042899/0.043282`, endpoint error
    `196.09/245.65/219.73/133.34 mm`.
  Neither arm reaches an endpoint or completes a traversal. Pretraining gives
  a better 16k start and prevents the control's final collapse, but the nearly
  identical 49k Dice means it does not simply solve deterministic navigation.
- PPO diagnostics are healthy and closely matched rather than showing a
  numerical failure. Both arms finish all 128 collector updates with maximum
  KL below `0.016`, last-ten value loss `0.365/0.410` (control/pretrained),
  stochastic recent-position uniqueness `0.9903/0.9894`, immediate reversal
  `0.00744/0.00626`, and zero logged off-target cliff. The pretrained policy's
  last-ten maximum action probability is only `0.01922` over 156 actions
  (`0.01331` control), so deterministic mode is still selected from a broad
  stochastic policy.
- Fixed stochastic seeds 101/202/303 on each arm's best deterministic
  checkpoint reveal the useful signal hidden by mode. Across nine
  development episodes, pretraining raises mean Dice
  `0.08130 -> 0.20682` and reduces endpoint error
  `156.04 -> 102.74 mm`. Every pretrained patient/seed Dice is
  `0.1281-0.2837`; recent-position uniqueness remains `0.9883`, boundary
  residence is only `0.0660`, and there are still zero endpoint/traversal
  successes. This is not revisit or boundary reward hacking, but it remains
  reward-supervised PPO training rather than annotation-free training.
- Test a likelihood-preserving deterministic decoder because the 156 actions
  factor into 26 directions and six lengths. It sums probabilities over
  lengths, selects the marginal direction, then its conditional MAP length;
  sampling, entropy, and every PPO log-probability are exactly unchanged.
  Reject it as the default: pretrained Dice falls `0.04328 -> 0.02936` and
  endpoint error worsens `133.34 -> 163.95 mm`. Ordinary joint mode remains
  the less-bad deterministic decoder.
- Resume the exact pretrained 65,536-frame checkpoint, including optimizer
  and frozen post-anneal learning rate, to 256,000 frames as
  `navigator-pretrained-development256k-v1.service`. Do not change the reward,
  action support, decoder, split, or seed during this promotion. Validation
  remains limited to the three development patients; pt14/pt18 stay sealed.
- The continuation restores `collected_frames=65,536`, `num_updates=640`,
  optimizer, and scheduler, then completes at 256,000 in `17m40s`.
  Deterministic joint-mode results are:
  - 102,400: Dice `0.027287`, endpoint `215.31 mm`;
  - 204,800: Dice `0.048121`, endpoint `117.52 mm`;
  - 256,000: Dice `0.037235`, endpoint `125.80 mm`.
  Every gate has zero endpoint/traversal success. The 204.8k best is only
  `0.00484` Dice above the 64k best and the final policy regresses. A single
  non-fatal 20-MiB expandable-segment mapping warning occurs during final
  validation with only 64 MiB free; the process completes and saves its
  checkpoint. Peak allocation/reservation is `12,830.7/14,678 MiB`.
- Fixed stochastic seeds on the 204.8k best also regress versus the 64k best:
  Dice `0.16332` versus `0.20682` and endpoint error `209.07` versus
  `102.74 mm`, still with zero success and `0.99045` position uniqueness.
  Therefore do not continue this unchanged objective toward one million
  frames merely because the stochastic 64k result was promising.
- Also test the categorical expected displacement projected to the nearest
  feasible lattice action, the Bayes action for squared displacement error.
  It exactly preserves sampling and likelihoods but lowers pretrained Dice to
  `0.02321`. Endpoint error improves to `91.48 mm` while recent-position
  uniqueness collapses to `0.03255`: conservative one-voxel consensus moves
  become locally trapped. Keep both alternative decoders audit-only and retain
  ordinary joint mode as the default.
- Conclusion: label-free masked image pretraining is a real representation and
  sample-efficiency improvement, especially for stochastic PPO
  (`0.0813 -> 0.2068` development Dice at the matched best checkpoints).
  It does not meet the `>=0.40` Dice or full-traversal goal, and longer PPO
  degrades the stochastic result. The next method should explicitly train a
  directional/sequence-consistent latent objective or sequence-level planner,
  not add reward scale, use GT as an input, or extend this checkpoint
  unchanged. The complete remote suite passes: `113 passed`, four subtests
  passed, 18 warnings.
