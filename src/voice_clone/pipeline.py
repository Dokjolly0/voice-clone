"""`status` (where each stage of a voice stands) and `run` (every stage in order, skipping finished ones)."""

from datetime import datetime

from voice_clone import audio, dataset, export, jobs, reference, train
from voice_clone.util import info, read_json, step
from voice_clone.workspace import Voice


def status(voice: Voice) -> dict:
    inputs = voice.inputs()
    ref = reference.status(voice)
    data = dataset.status(voice)
    prog = train.progress(voice)
    target = read_json(voice.training_dir / "run.json", {}).get("epochs")
    onnx = voice.onnx_path
    return {
        "name": voice.name,
        "language": voice.language.name,
        "inputs": len(inputs),
        "input_seconds": sum(audio.duration(p) for p in inputs),
        "reference": ref,
        "sentences": len(voice.sentences()),
        "dataset": data,
        "training": {**prog, "target_epochs": target, "last": str(prog["last"]) if prog["last"] else None},
        "exported": datetime.fromtimestamp(onnx.stat().st_mtime).isoformat(timespec="minutes")
        if onnx.exists() else None,
        "job": jobs.current(voice),
    }


def format_status(s: dict) -> str:
    def mark(done: bool) -> str:
        return "[x]" if done else "[ ]"

    ref, data, tr = s["reference"], s["dataset"], s["training"]
    lines = [f"{s['name']} ({s['language']})"]
    lines.append(f"  {mark(s['inputs'] > 0)} 1. audio      {s['inputs']} files, {s['input_seconds']:.1f} s")
    if ref["reference_seconds"]:
        ref_text = (f"{ref['reference_seconds']:.1f} s, transcript {'yes' if ref['transcript'] else 'no'}, "
                    f"voice {'yes' if ref['voice'] else 'no'}")
    else:
        ref_text = "not built"
    lines.append(f"  {mark(ref['voice'])} 2. reference  {ref_text}")
    if s["sentences"]:
        ds_text = (f"{s['sentences']} sentences: {data['verified']} good clips, {data['pending']} to do, "
                   f"{data['failed']} failed; {data['exported']} exported")
    else:
        ds_text = "no sentences yet"
    lines.append(f"  {mark(data['exported'] > 0)} 3. dataset    {ds_text}")
    if tr["epoch"] is not None:
        epoch = f"epoch {tr['epoch'] + 1}" + (f"/{tr['target_epochs']}" if tr["target_epochs"] else "")
        best = f", best val_mel {tr['best_val_mel']:.4f} at epoch {tr['best_epoch']}" if tr["best_val_mel"] else ""
        tr_text = epoch + best + (" (finished)" if tr["finished"] else "")
    else:
        tr_text = "not started"
    lines.append(f"  {mark(tr['finished'])} 4. training   {tr_text}")
    lines.append(f"  {mark(bool(s['exported']))} 5. export     {s['exported'] or 'not exported'}")
    if s["job"]:
        lines.append(f"  running: voice-clone {' '.join(s['job']['args'])} (pid {s['job']['pid']}, "
                     f"since {s['job']['started']})")
    return "\n".join(lines)


def run(voice: Voice, *, max_seconds: float, sentence_count: int, extra, dataset_settings: dataset.Settings,
        train_settings: train.Settings, checkpoint: str) -> None:
    """Every stage in order. Finished stages are skipped; an interrupted training run is resumed."""
    step("Stage 2/5: reference voice")
    reference.build(voice, max_seconds=max_seconds)

    step("Stage 3/5: dataset")
    from voice_clone import sentences

    sentences.choose(voice, sentence_count, extra=extra)
    dataset.build(voice, dataset_settings)

    step("Stage 4/5: training")
    prog = train.progress(voice)
    target = read_json(voice.training_dir / "run.json", {}).get("epochs")
    if prog["finished"] and target == train_settings.epochs:
        info(f"training already finished ({target} epochs)")
    else:
        # Interrupted, or finished with fewer epochs than asked now: continue from the last checkpoint.
        train_settings.resume = prog["last"] is not None
        train.run(voice, train_settings)

    step("Stage 5/5: export and test")
    export.export_and_test(voice, checkpoint=checkpoint)
