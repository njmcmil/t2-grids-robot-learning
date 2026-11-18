from pathlib import Path
import hashlib, json
from .models import VARIANTS
from .train import load_policy
from .simulation import rollout


def freeze_selection(root, checkpoint_paths, success_target, justification):
    root = Path(root)
    protocol_path = root / "configs/evaluation_protocol.json"
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol["success_target"] != success_target or not justification.strip():
        raise ValueError(
            "Target must match the protocol and have a written justification before testing."
        )

    # Record exactly which saved model represents each variant and training seed.
    selections = []
    seen = set()
    for path in checkpoint_paths:
        path = Path(path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Keep checkpoints inside the project.")
        _, saved, _ = load_policy(path, "cpu")
        cfg = saved["config"]
        if cfg["smoke_test"]:
            raise ValueError("Smoke-test checkpoint is not eligible.")
        pair = (cfg["variant"], cfg["training_seed"])
        if pair in seen:
            raise ValueError("Duplicate variant/seed")
        seen.add(pair)
        assert cfg["manifest_sha256"] == protocol["split_manifest_sha256"]
        selections.append(
            {
                "path": str(path.relative_to(root.resolve())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "variant": pair[0],
                "training_seed": pair[1],
                "selected_epoch": saved["epoch"],
                "validation_mse": saved["validation_mse"],
                "budget": [cfg["epochs"], cfg["batch_size"], cfg["learning_rate"]],
            }
        )
    expected = {(v, s) for v in VARIANTS for s in (0, 1)}
    if seen != expected:
        raise ValueError(
            f"Need one selected checkpoint for each variant and seed 0/1. Missing: {expected-seen}"
        )
    if len({tuple(s["budget"]) for s in selections}) != 1:
        raise ValueError(
            "Unequal training budgets: resolve or explicitly revise protocol before final comparisons."
        )
    result = {
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "success_target": success_target,
        "target_justification": justification,
        "checkpoints": selections,
        "selection_rule": "Lowest validation MSE within each chosen run; choose runs using validation only.",
    }
    path = root / "configs/final_selection.json"
    if path.exists():
        raise FileExistsError(
            "Final selection already frozen. Do not overwrite after viewing test outcomes."
        )
    path.write_text(json.dumps(result, indent=2))
    return path


# Run each selected model on the same starting scenes and reuse completed results.
def evaluate_selected(root):
    root = Path(root)
    selection = json.loads((root / "configs/final_selection.json").read_text())
    raw = (root / "configs/evaluation_protocol.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == selection["protocol_sha256"]
    protocol = json.loads(raw)
    for selected in selection["checkpoints"]:
        checkpoint = root / selected["path"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected["sha256"]
        out = (
            root
            / "results/final"
            / f"{selected['variant']}_seed{selected['training_seed']}"
        )
        for case in protocol["cases"]:
            destination = out / (case["case_id"] + ".json")
            if destination.exists():
                previous = json.loads(destination.read_text())
                assert (
                    previous["protocol_sha256"] == digest
                    and previous["checkpoint_sha256"] == selected["sha256"]
                )
                continue
            result = rollout(
                root,
                checkpoint,
                case,
                out,
                timeout=protocol["timeout_steps"],
                video=True,
                protocol_hash=digest,
            )

    return root / "results/final"
