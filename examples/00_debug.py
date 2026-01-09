import sys
import omnigibson as og
from realm.eval_vectorized import evaluate


if __name__ == "__main__":
    evaluate(
        task_id=6,
        perturbation_id=0,
        repeats=3,
        num_envs=3,
        max_steps=45,
        model_type="debug",
        port=8000
    )
    og.shutdown()
    sys.exit(0)
