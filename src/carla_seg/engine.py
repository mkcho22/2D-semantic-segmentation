"""Training, safe checkpoint loading, evaluation, and prediction panels."""

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from .classes import CLASS_COLORS, CLASS_NAMES, NUM_CLASSES
from .data import CarlaDataset
from .metrics import SegmentationMetrics, hybrid_loss
from .model import UNet


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(weights, device):
    """Accept original state_dict files and this repository's checkpoint format."""
    payload = torch.load(weights, map_location="cpu", weights_only=True)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model = UNet(n_classes=NUM_CLASSES)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def make_dataset(args, split):
    return CarlaDataset(args.data_dir, split, args.height, args.width, args.skip_missing)


def make_loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator)


@torch.inference_mode()
def evaluate_model(model, loader, device, ce_weight=0.5):
    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES)
    total_loss = 0.0
    samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = hybrid_loss(logits, labels, ce_weight)
        if not torch.isfinite(loss):
            raise ValueError("Evaluation produced a non-finite loss")
        total_loss += loss.item() * len(images)
        samples += len(images)
        metrics.update(logits, labels)
    report = metrics.compute()
    report.update(loss=total_loss / samples, samples=samples)
    return report


def train(args):
    seed_everything(args.seed)
    device = select_device(args.device)
    train_data = make_dataset(args, "Training")
    validation_data = make_dataset(args, "Validation")
    output = Path(args.output_dir)
    if any((output / name).exists() for name in ("best.pt", "last.pt", "history.csv")):
        raise ValueError(f"Training output already exists in {output}; choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(device=str(device), num_classes=NUM_CLASSES,
                  loss="ce_plus_multiclass_dice", dropout=0.5,
                  torch_version=str(torch.__version__),
                  training_data=train_data.report, validation_data=validation_data.report)
    save_json(output / "config.json", config)
    train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = make_loader(validation_data, args.batch_size)
    model = UNet(n_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_loss, stale_epochs = float("inf"), 0
    fields = ("epoch", "train_loss", "val_loss", "val_miou", "val_miou_percent", "seconds")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            model.train()
            total_loss, samples = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
                with context:
                    logits = model(images)
                    loss = hybrid_loss(logits, labels, args.ce_weight)
                if not torch.isfinite(loss):
                    raise ValueError("Training produced a non-finite loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(images)
                samples += len(images)
            validation = evaluate_model(model, validation_loader, device, args.ce_weight)
            row = dict(epoch=epoch, train_loss=total_loss / samples,
                       val_loss=validation["loss"], val_miou=validation["miou"],
                       val_miou_percent=validation["miou_percent"],
                       seconds=time.monotonic() - start)
            writer.writerow(row)
            history_file.flush()
            checkpoint = {
                "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "epoch": epoch, "config": config, "validation": validation,
            }
            torch.save(checkpoint, output / "last.pt")
            if validation["loss"] < best_loss:
                best_loss, stale_epochs = validation["loss"], 0
                torch.save(checkpoint, output / "best.pt")
                save_json(output / "best_metrics.json", validation)
            else:
                stale_epochs += 1
            print(json.dumps(row), flush=True)
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best validation loss: {best_loss:.6f}")
                break
    print(f"Saved best.pt, last.pt, config.json, history.csv to {output}")


def evaluate(args):
    device = select_device(args.device)
    data = make_dataset(args, args.split)
    selected = Subset(data, range(min(args.limit, len(data)))) if args.limit else data
    model = load_model(args.weights, device)
    start = time.monotonic()
    report = evaluate_model(model, make_loader(selected, args.batch_size), device, args.ce_weight)
    report.update(
        weights_name=Path(args.weights).name, split=args.split,
        input_height=args.height, input_width=args.width, batch_size=args.batch_size,
        device=str(device), available_pairs=len(data), subset=len(selected) != len(data),
        sample_selection="first N sorted exact-name pairs" if args.limit else "all exact-name pairs",
        data_report=data.report, class_names=list(CLASS_NAMES),
        elapsed_seconds=time.monotonic() - start,
    )
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


@torch.inference_mode()
def predict(args):
    device = select_device(args.device)
    data = make_dataset(args, args.split)
    model = load_model(args.weights, device)
    count = min(args.count, len(data))
    # Match the portfolio preview: start of each equal-sized dataset segment.
    indices = [i * len(data) // count for i in range(count)]
    tile_width = min(480, args.width)
    tile_height = round(tile_width * args.height / args.width)
    margin, header, row_title = 16, 48, 28
    panel = Image.new("RGB", (tile_width * 3 + margin * 4,
                              header + count * (tile_height + row_title + margin)), "#101827")
    draw = ImageDraw.Draw(panel)
    for column, title in enumerate(("RGB input", "U-Net prediction", "Ground truth")):
        draw.text((margin + column * (tile_width + margin), 18), title, fill="white")
    names = []
    for row, index in enumerate(indices):
        image_tensor, label = data[index]
        prediction = model(image_tensor.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        image_path = data.pairs[index][0]
        names.append(image_path.name)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((args.width, args.height), Image.Resampling.BILINEAR)
        panels = (rgb, Image.fromarray(CLASS_COLORS[prediction]), Image.fromarray(CLASS_COLORS[label.numpy()]))
        top = header + row * (tile_height + row_title + margin)
        draw.text((margin, top), image_path.name, fill="#cbd5e1")
        for column, image in enumerate(panels):
            resized = image.resize((tile_width, tile_height),
                                   Image.Resampling.BILINEAR if column == 0 else Image.Resampling.NEAREST)
            panel.paste(resized, (margin + column * (tile_width + margin), top + row_title))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    metadata = dict(weights_name=Path(args.weights).name, split=args.split,
                    filenames=names, input_height=args.height, input_width=args.width,
                    device=str(device), data_report=data.report,
                    note="Deterministic evenly spaced sample; qualitative panel, not a performance benchmark.")
    save_json(output.with_suffix(".json"), metadata)
    print(f"Saved {output} and {output.with_suffix('.json')}")
