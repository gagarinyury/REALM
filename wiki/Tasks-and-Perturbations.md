# Tasks and Perturbations

The benchmark axis is `task × perturbation`: 10 tasks × 16 perturbations = 160 cells,
each typically run for 25 repeats at 800 steps in the paper. This page is the
authoritative reference for both axes — IDs, names, what's actually under the hood,
and where the YAML lives.

## Tasks (`--task_id`)

`SUPPORTED_TASKS` is defined in `realm/eval.py`. The string is also the directory name
under `realm/config/tasks/REALM_DROID10/<task>/default.yaml`.

| ID | Task | Skill type | Main / target objects |
| :-- | :--- | :--- | :--- |
| 0 | `put_green_block_into_bowl` | `put` | green cube → bowl |
| 1 | `put_banana_into_box` | `put` | banana → box |
| 2 | `rotate_marker` | `rotate` | marker |
| 3 | `rotate_mug` | `rotate` | mug |
| 4 | `pick_spoon` | `pick` | spoon |
| 5 | `pick_water_bottle` | `pick` | water bottle |
| 6 | `stack_cubes` | `stack` | cube → cube |
| 7 | `push_switch` | `push` | wall switch (uses `ToggledOn` state) |
| 8 | `open_drawer` | `open_drawer` | drawer handle |
| 9 | `close_drawer` | `close_drawer` | drawer handle |

Note the spelling discrepancy with the README table:
`put_green_block_in_bowl` (README) vs. `put_green_block_into_bowl` (code & dirs). The
**code spelling** is canonical; always use the ID.

### Skill-compatibility matrix

Used by `SB-VRB` (verb perturbation) to substitute in a compatible verb without
breaking the scene. Defined in `RealmEnvironmentDynamic.SKILL_COMPATIBILITY_MATRIX`:

| From | Compatible substitutes |
| :--- | :--- |
| `put` | `pick`, `rotate`, `stack` |
| `pick` | `put`, `rotate`, `stack` |
| `rotate` | `put`, `pick`, `stack` |
| `stack` | `put`, `pick`, `rotate` |
| `push` | *(none — `SB-VRB` on push is a no-op)* |
| `open` | `close` |
| `close` | `open` |

### Task config layout

Each task lives under `realm/config/tasks/REALM_DROID10/<task>/default.yaml` and
declares:

- `task_type` (used for skill matching)
- `instruction` and the placeholder spans (`instruction_obj_to_replace`, etc.)
- `supported_scenes` — `Pomaria_1_int: ["Table"]` for all REALM_DROID10 tasks
- `camera_extrinsics` — references named entries in `config/env/external_sensors/camera_extrinsics.yaml`
- `reset_joint_pos` — DROID start configuration
- `main_objects`, `target_objects`, `distractors`, optional `immutables`
- `cached_semantic_perturbations` — pre-generated paraphrases for `S-LANG`, `S-AFF`,
  `S-INT`, `S-PROP`, `S-MO` (sampled from at runtime)

Stage rubrics for graded `task_progression` live in
`realm/config/tasks/task_progressions.yaml`.

## Perturbations (`--perturbation_id`)

`SUPPORTED_PERTURBATIONS` in `realm/eval.py`. Implementations live under
`realm/environments/perturbations/`.

| ID | Name | Category | What it perturbs | File |
| :-- | :--- | :--- | :--- | :--- |
| 0 | `Default` | — | nothing | `default.py` |
| 1 | `V-AUG` | Visual | per-frame Gaussian blur σ ∈ [0,3], contrast α ∈ [0.5, 2.0] | applied at obs time |
| 2 | `V-VIEW` | Visual | external camera pose (mixed rotation noise) | `v_view.py` |
| 3 | `V-SC` | Visual | spawns 3 random distractors | `v_sc.py` |
| 4 | `V-LIGHT` | Visual | light color & intensity | `v_light.py` |
| 5 | `S-PROP` | Semantic | swaps to a property-based instruction | `semantic.py::s_prop` |
| 6 | `S-LANG` | Semantic | swaps to a paraphrase / verb-substitute | `semantic.py::s_lang` |
| 7 | `S-MO` | Semantic | references spatial relations | `semantic.py::s_mo` |
| 8 | `S-AFF` | Semantic | references human affordance / use case | `semantic.py::s_aff` |
| 9 | `S-INT` | Semantic | references world knowledge ("plastic toy", etc.) | `semantic.py::s_int` |
| 10 | `B-HOBJ` | Behavioral | randomizes manipulated-object mass | `b_hobj.py` |
| 11 | `SB-NOUN` | Sem + Beh | swaps the noun for another in-scene object | `sb_noun.py` |
| 12 | `SB-VRB` | Sem + Beh | swaps the verb for a compatible one | `sb_vrb.py` |
| 13 | `VB-POSE` | Vis + Beh | randomizes manipulated-object pose | `vb_pose.py` |
| 14 | `VB-MOBJ` | Vis + Beh | randomizes object size & shape | `vb_mobj.py` |
| 15 | `VSB-NOBJ` | Vis + Sem + Beh | samples an unseen manipulated object | `vsb_nobj.py` |

Categories follow the paper:
- **V-Avg.** averages 1–4 (V-*)
- **S-Avg.** averages 5–9 (S-*)
- **B-Avg.** averages 10, 13, 14 (B-* and *B-* without semantics)

Perturbations marked but not yet shipped: `V-OBJ`, `VB-ISC`, `VS-PROP`, `SB-ADV`,
`SB-SMO` are in `MISSING_PERTURBATIONS` and intentionally not exposed.

### Compatibility constraints

- `SB-NOUN` × `push` (perturbation 11 × task 7) raises `NotImplementedError` in
  `RealmEnvironmentDynamic.__init__`. Skip this cell when sweeping.
- `VSB-NOBJ` shifts the main object's `z` by +0.3 m for `open_drawer` and `close_drawer`
  (compensates for unseen-object spawn collisions with the drawer geometry).
- `V-AUG` does not modify the env at construction; it's applied per-frame to the
  observed image inside `extract_from_obs`.

## Running a custom task config

If you have a custom YAML (e.g. a per-camera variant), pass it explicitly:

```bash
python /app/examples/02_evaluate.py \
    --task_cfg_path REALM_DROID10/rotate_mug/default.yaml \
    --perturbation_id 4 \
    --model_type openpi --port 8000 \
    --model_name pi05 --experiment_name custom_run
```

The script accepts a path **relative to `realm/config/tasks/`**. Filenames other than
`default.yaml` are appended to the task name in logs, so
`REALM_DROID10/rotate_mug/heavy.yaml` reports under `rotate_mug_heavy`.
