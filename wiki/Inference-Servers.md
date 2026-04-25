# Inference Servers

`02_evaluate.py` does not run a model itself — it expects a policy server reachable on
`--host:--port` and queries it once per action chunk. This page lists the supported
`--model_type`s, their wire formats, and how to start each server.

## Supported `--model_type` values

The full set is implemented in `realm/inference/client.py::InferenceClient`:

| `--model_type` | Server / library | Wire format | Notes |
| :--- | :--- | :--- | :--- |
| `openpi` | [openpi](https://github.com/Physical-Intelligence/openpi) `serve_policy.py` | `WebsocketClientPolicy` | Used by Pi0 / Pi0.5 / Pi0-FAST checkpoints. Default for the README quickstart. |
| `GR00T` | NVIDIA GR00T `serve_gr00t.py` (zmq) | `client.infer({...})` | Resizes to 320×180, sends two exterior cams + wrist + joint pos + gripper. |
| `GR00T_N16` | GR00T N1.6 server | `client.get_action({...})` | Resizes to 224×224, single exterior cam + wrist. Disabled by default in the import block — uncomment the relevant lines in `client.py` to enable. |
| `molmoact` | molmoact server | `client.infer({...})` | One exterior + wrist image, one instruction. Gripper polarity is **inverted** (see below). |
| `hamster` | Hamster trajectory server | `client.infer(img_bgr, instruction)` | Disabled by default (commented import). Sends a BGR image and a string. |
| `dreamzero` | DreamZero server | `client.infer({...})` | Disabled by default. Requires `--multi-view` and the robot-relative cartesian EE pose. |
| `debug` | None | — | Returns a constant action vector. Use it to time the simulator independently of any model. |

> The commented imports at the top of `client.py` (`HamsterClient`, `DreamZeroClient`,
> `ExternalRobotInferenceClient`) need to be uncommented before those `model_type`s
> work end-to-end. They were turned off so the eval can run inside the slim
> `omnigibson` micromamba env without pulling extra deps.

## Gripper binarization

`realm/eval.py` post-processes `action[-1]` based on `model_type`:

```python
if model_type in ["debug", "openpi", "GR00T", "GR00T_N16", "dreamzero"]:
    new_action[-1] = 1 if action[-1] > 0.5 else -1
elif model_type == "molmoact":
    new_action[-1] = 1 if action[-1] < 0.5 else -1
else:
    raise NotImplementedError()
```

Two consequences:
1. **Adding a new model type to the eval requires editing this block.** Otherwise
   `evaluate` raises `NotImplementedError` on the very first step.
2. **molmoact's polarity is inverted.** If you fork molmoact and your model emits the
   `(open=1, closed=0)` convention instead of `(open=0, closed=1)`, you'll need to flip
   it back.

## Starting an `openpi` server

Local workstation:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
uv sync
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_full_droid_finetune \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

Default port is `8000`. Pass `--port` to bind a different one.

Cluster (Apptainer, what `run_single_eval.sh` does internally):

```bash
apptainer exec \
    --writable-tmpfs --nv \
    --bind "$(pwd)":/app \
    --bind "$CHECKPOINT":/checkpoint \
    --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    --env GIT_LFS_SKIP_SMUDGE=1 \
    "$OPENPI_SIF" uv run /app/scripts/serve_policy.py \
        --port=$PORT \
        policy:checkpoint \
        --policy.config=$POLICY_CONFIG \
        --policy.dir=/checkpoint
```

Tune `XLA_PYTHON_CLIENT_MEM_FRACTION` if you co-locate the policy and the simulator on
the same GPU — `0.25` is a safe ceiling for that case; `0.5` is fine on a dedicated
policy GPU.

## Starting a `GR00T` server

```bash
cd $GR00T_ROOT
uv run scripts/serve_gr00t.py \
    --port=$PORT \
    --model_path /path/to/checkpoint \
    --data-config droid_joint_pos
```

`02_evaluate.py --model_type GR00T` resizes both exterior cameras and the wrist camera
to 320×180 before sending. Make sure your finetune was trained at that resolution.

## Starting a `molmoact` server

The cluster wrapper in `run_single_eval.sh` invokes molmoact through a separate
Apptainer image. The relevant snippet:

```bash
apptainer exec --nv --writable-tmpfs \
    --bind "$(pwd)":/app \
    --bind "$CHECKPOINT_PATH":/checkpoint \
    "$MOLMOACT_SIF" /bin/bash -c \
    "source /opt/conda/etc/profile.d/conda.sh && conda activate \
     && pip install tyro \
     && pip install /app/packages/openpi-client \
     && python /app/inference/run_molmoact_server.py --port=$PORT"
```

`run_molmoact_server.py` is **not vendored** in this repo — point the `--bind` at the
molmoact codebase that hosts it (the path baked into the cluster script
`/scratch/.../molmoact/apptainer/molmoact.sif` is author-specific).

## `dreamzero` requirements

`dreamzero` is the strictest about its observation:

- **Must** pass `--multi-view`. The client asserts `base_im_second is not None`.
- **Must** have a working `cartesian_position` — `realm/eval.py` computes this from the
  current EE pose via `env._world2robot`.
- Images are resized to 320×180 with `np.uint8` dtype, NOT padded.

If you see `assert base_im_second is not None`, you forgot `--multi-view`.

## `debug` mode

```bash
python /app/examples/02_evaluate.py \
    --model_type debug --model_name debug \
    --task_id 0 --perturbation_id 0 \
    --port 0 \
    --experiment_name profile_sim
```

Returns a fixed 8-D action vector (or a 6-D EE pose when `ee_control=True`). Useful for
profiling Isaac Sim without dragging in a model GPU. Combine with `--no_render` to
measure the physics step time alone.

## Adding a new model

1. Add a branch to `InferenceClient.infer` that builds the right `obs_dict` and calls
   the server.
2. Optionally add a connection branch to `InferenceClient.__init__` (e.g. to wrap a
   custom client class).
3. Add the new `model_type` string to the gripper-binarization block in
   `realm/eval.py`.
4. If your model needs the second exterior camera, set `use_base_im_second=True` in the
   call site (it's currently inferred from `env.task_type == "open_close_drawer"`).
5. Document the wire format here.

The cluster scripts only need a new branch in `run_single_eval.sh` if you want sbatch
to launch the server for you. Otherwise start it on a head node and point
`--host`/`--port` at it.
