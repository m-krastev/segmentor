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
