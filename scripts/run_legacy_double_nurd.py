"""Pinned b5b2272 Mequinna reproduction plus an additive tail-QCD critic.

Run as a standalone FILE, in a fresh process: imports of the original recipe
are isolated from the modern training modules. Never writes original artifacts.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint

LEGACY_COMMIT = "b5b2272939cfe8898c7001c5b3abfcfb8c94ef4e"
PROTOCOL = "golden_mequinna_additive_tail_nurd_v1"
PINNED = {
    "train_hlt.py": "f05ccb1207ec6a4aacf86fb873932831060b4e94efe14062f761d7f3f1b19866",
    "dataset/hlt_smcocktail_dataset.py": "ca0d9bd8870662bf55b4410712f7422e95f550f951143d6eaa676078e25cb1d0",
    "models/hlt_con.py": "1f4a46fa0466d2529b6ea28c63bfa8641784bc22fdf46d5341533370e2411670",
    "models/hlt_autoencoder.py": "217baf74c2b1ddda4230ca872b9ee4c5e0e7c88ef57aeb8ff283a1d74b7a4982",
    "utils/hlt_weights.py": "67b1392bdd20a6131c311b27122bc8a05bc84495a5a9ade861ea53e7fe620ea0",
    "utils/hlt_density_ratio.py": "dc761e7d13a9df22db937cb45b9c25d34e9c7fd48e12c0a2b9a49c0e6e41d62b",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): result.update(block)
    return result.hexdigest()


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.generic): x = x.item()
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def pinned_recipe(path):
    path = path.resolve()
    for name, expected in PINNED.items():
        if sha(path/name) != expected: raise ValueError(f"Pinned recipe modified: {name}")
    if any(name in sys.modules for name in ("train_hlt", "utils.hlt_weights", "models.hlt_con")):
        raise RuntimeError("Legacy driver must run in a fresh standalone Python process")
    sys.path.insert(0, str(path))
    import train_hlt as recipe
    if Path(recipe.__file__).resolve() != path/"train_hlt.py": raise ValueError("Wrong trainer imported")
    return recipe


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"]); random.setstate(state["python"])


@contextmanager
def isolated_rng(seed):
    state = rng_state()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: yield
    finally: restore_rng(state)


@contextmanager
def evaluation_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try: yield
    finally:
        for module, mode in modes: module.training = mode


def checkpoint_blocks(model):
    """Same batch/loss/model; recompute Transformer activations to save VRAM.

    There is no BN in these blocks. RNG preservation replays dropout without
    advancing the enclosing training RNG. Model state-dict keys do not change.
    """
    for block in model.layers:
        original = block.forward
        def forward(x, mask=None, original=original, block=block):
            if block.training and torch.is_grad_enabled():
                return checkpoint(original, x, mask, use_reentrant=False, preserve_rng_state=True)
            return original(x, mask)
        block.forward = forward


def make_model(recipe, config, tokens, device):
    keys = ("embed_size", "latent_dim", "proj_dim", "num_heads", "num_layers", "dim_ff", "linear_dim", "dropout")
    return recipe.HLTContrastiveModel(num_classes=4, num_tokens=tokens, **{k: config[k] for k in keys}).to(device)


def make_dataset(recipe, pf, labels, scores, physics, pre, indices, effective):
    from dataset.hlt_smcocktail_dataset import HLTSmCocktailDataset
    class Safe(HLTSmCocktailDataset):
        def __getitem__(self, item):
            batch = super().__getitem__(item)
            return (torch.nan_to_num(batch[0].float(), nan=0., posinf=0., neginf=0.), *batch[1:])
    nuisance = (scores-float(pre["nuisance_transform"]["mean"])) / float(pre["nuisance_transform"]["std"])
    return Safe(pf, labels, nuisance, scores, effective, physics, indices, "legacy_experiment")


def score_ae(model, scaler, obj, device, batch_size):
    mu, std = scaler["mu"].cpu().float(), scaler["std"].cpu().float()
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    scores = []
    with torch.no_grad():
        for start in range(0, len(obj), batch_size):
            x = torch.nan_to_num(obj[start:start+batch_size, :, :4].reshape(-1, mu.numel()).float(),
                                 nan=0., posinf=0., neginf=0.)
            x = ((x-mu)/std).to(device)
            reco, _ = model(x)
            scores.append((reco-x).square().mean(1).cpu())
    return torch.cat(scores)


def prepare(args, recipe):
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    device = torch.device(args.device)
    source = torch.load(args.golden_checkpoint, map_location="cpu", weights_only=False)
    ae_hash = sha(args.ae_checkpoint)
    cfg = source["config"]
    if (cfg["code_commit"] != LEGACY_COMMIT or cfg["contrast_weight"] != .3
            or cfg["lambda_info"] != 1. or cfg["balance_strata"] != 20
            or source["state_dict_model"]["classifier.weight"].shape[0] != 4
            or source["ae_checkpoint_sha256"] != ae_hash):
        raise ValueError("Expected the verified original b5b2272 parent4/SupCon=.3/lambda=1 checkpoint and matching AE")
    config = vars(recipe.build_parser().parse_args(["--data", str(args.sample), "--ae_ckpt", str(args.ae_checkpoint)]))
    config.update(cfg)
    if config.get("exclude_labels") or config.get("max_events", -1) > 0:
        raise ValueError("Expected complete original parent4 population")
    pre = source["preprocessing"]
    from utils.hlt_weights import apply_joint_balance, load_generator_weights, sample_signature, weighted_quantile, stratified_split_indices
    raw = torch.load(args.sample, map_location="cpu", weights_only=False)
    if sample_signature(raw) != pre["data_signature"]: raise ValueError("Original training manifest differs")
    labels = raw["label"].long().reshape(-1)
    if set(labels.tolist()) != set(range(4)): raise ValueError("Native parent4 data required")
    physics, metadata = load_generator_weights(str(args.weights), labels, qcd_label=1, sample=raw)
    if metadata["effective_physics_weight_sha256"] != pre["weighting"]["generator"]["effective_physics_weight_sha256"]:
        raise ValueError("Legacy generator weights changed")
    train, cal = (pre["split"][key].long() for key in ("train_indices", "validation_indices"))
    combined = torch.cat((train, cal))
    if len(combined) != len(labels) or not torch.equal(torch.sort(combined).values, torch.arange(len(labels))):
        raise ValueError("Saved train/calibration rows must partition the original file")
    recipe.set_random_seed(config["manualSeed"])
    ae_model, ae = recipe.load_ae_checkpoint(str(args.ae_checkpoint), device)
    recipe.validate_ae_training_contract(ae, pre)
    if ae["weighting"]["method"] != "generator_weighted_uniform_class":
        raise ValueError("Original class-balanced AE required for this historical reproduction")
    if (ae["weighting"]["split_seed"] != pre["split"]["seed"]
            or ae["weighting"]["validation_fraction"] != pre["split"]["validation_fraction"]):
        raise ValueError("AE-unseen calibration membership is not the original shared split")
    ae_train, ae_cal = stratified_split_indices(labels, val_fraction=config["val_split"],
        seed=config["manualSeed"], weights=physics)
    if not torch.equal(ae_train, train) or not torch.equal(ae_cal, cal):
        raise ValueError("Pinned reconstruction of AE/classifier split differs")
    scores = score_ae(ae_model, source["ae_scaler"], raw["obj"], device, config["ae_batch_size"])
    balance = pre["weighting"]["balance_spec"]
    if balance.get("version") != 1: raise ValueError("Expected original un-clipped weighted-quantile stratum specification")
    train_effective = apply_joint_balance(labels[train], scores[train], physics[train], balance)
    cal_effective = apply_joint_balance(labels[cal], scores[cal], physics[cal], balance)
    threshold = float(weighted_quantile(scores[cal], torch.tensor([.9]), physics[cal])[0])
    tail = train[(labels[train] == 1) & (scores[train] >= threshold)]
    if len(tail) < 2 or physics[tail].sum() <= 0: raise ValueError("No trainable positive-mass tail QCD")
    # Original seed -> AE construction -> encoder/global-critic construction.
    model = make_model(recipe, config, raw["pf"].shape[1], device)
    critic = recipe.HLTCritic(config["latent_dim"], 4).to(device)
    initial = {"state_dict_model": model.state_dict(), "state_dict_critic": critic.state_dict(), "rng": rng_state()}
    args.output_dir.mkdir(parents=True)
    torch.save(initial, args.output_dir/"matched_initial_state.pth.tar")
    model.load_state_dict(source["state_dict_model"]); critic.load_state_dict(source["state_dict_critic"])
    dataset = make_dataset(recipe, raw["pf"], labels, scores, physics, pre, cal, cal_effective)
    reproduction = recipe.validate(model, critic, DataLoader(dataset, batch_size=config["batch_size"], shuffle=False), device)
    expected = source["validation_metrics"]
    for key in ("weighted_ce", "weighted_accuracy", "balanced_accuracy"):
        if abs(reproduction[key]-expected[key]) > args.reproduction_tolerance:
            raise ValueError(f"Golden validation reproduction failed: {key}={reproduction[key]} versus {expected[key]}; dependent training will not start")
    for path, name in ((args.golden_checkpoint, "golden_checkpoint.pth.tar"), (args.ae_checkpoint, "original_checkpoint_ae.pth")):
        shutil.copy2(path, args.output_dir/name)
        if sha(path) != sha(args.output_dir/name): raise ValueError("Snapshot verification failed")
    prepared = {"config": config, "preprocessing": pre, "ae_scaler": source["ae_scaler"],
        "ae_checkpoint_sha256": ae_hash, "scores": scores, "physics": physics,
        "train_effective": train_effective, "cal_effective": cal_effective,
        "tail_QCD_indices": tail, "ae_threshold": threshold, "tokens": raw["pf"].shape[1],
        "data_signature": pre["data_signature"], "weight_metadata": metadata}
    torch.save(prepared, args.output_dir/"prepared_recipe.pth.tar")
    report = {"protocol": PROTOCOL, "legacy_commit": LEGACY_COMMIT, "code_commit": args.code_commit,
        "golden_sha256": sha(args.golden_checkpoint), "AE_sha256": ae_hash,
        "prepared_sha256": sha(args.output_dir/"prepared_recipe.pth.tar"),
        "initial_sha256": sha(args.output_dir/"matched_initial_state.pth.tar"),
        "sample": str(args.sample), "weights": str(args.weights), "config": config,
        "golden_validation": reproduction, "archived_validation": expected,
        "reproduction_tolerance": args.reproduction_tolerance, "ae_threshold": threshold,
        "train_events": len(train), "calibration_events": len(cal), "tail_QCD_events": len(tail),
        "tail_QCD_effective_events": float(physics[tail].sum().square()/physics[tail].square().sum()),
        "tail_measure": "AE-gradient-unseen calibration all-background LEGACY generator-weighted 90th percentile",
        "legacy_weights": "Original QCD weights; non-QCD unit base weights, NOT new all-process cross-section normalization",
        "AE_training": "None; original class-balanced historical AE reused unchanged",
        "caveats": ["No composite legacy event IDs; file signatures and saved row roles cannot prove no historical cross-file duplicate events",
            "Previously inspected MC; not a blind test", "AE calibration also selected the AE epoch",
            "Fresh seed-matched training is not guaranteed bitwise identical across hardware/library versions or historical W&B RNG side effects"]}
    write_json(args.output_dir/"preparation_complete.json", report)
    print("GOLDEN VALIDATION REPRODUCED; paired initialization and original weighting/split frozen.", flush=True)


class TailStream:
    def __init__(self, dataset, weights, batch_size, seed):
        self.dataset, self.weights, self.batch_size = dataset, weights.double(), batch_size
        self.generator = torch.Generator().manual_seed(seed); self.iterator = None
    def __next__(self):
        if self.iterator is None:
            sampler = WeightedRandomSampler(self.weights, self.batch_size*128, replacement=True, generator=self.generator)
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                                            num_workers=0, generator=self.generator))
        try: return next(self.iterator)
        except StopIteration: self.iterator = None; return next(self)


def tail_coefficient(epoch, warmup, ramp, maximum):
    if epoch <= warmup: return 0.
    return maximum*min(1., (epoch-warmup)/ramp) if ramp else maximum


def train_double_epoch(recipe, model, critic, tail_critic, loader, stream, optimizer, critic_opt,
                       tail_opt, device, cfg, contrastive, coefficient, epoch):
    metrics = recipe.EpochMetrics(4)
    critic_iterator = iter(loader)
    tail_metrics = []
    for step, batch in enumerate(loader):
        for _ in range(cfg.critic_steps):
            try: critic_batch = next(critic_iterator)
            except StopIteration: critic_iterator = iter(loader); critic_batch = next(critic_iterator)
            loss, accuracy = recipe.critic_update(model, critic, critic_opt, critic_batch, device)
            metrics.update_critic(loss, accuracy)
        # Extra sampling/RNG never advances the historical global data/dropout RNG.
        with isolated_rng(cfg.manualSeed + epoch*100000 + step):
            tail_batch = recipe._move_batch(next(stream), device)
            with evaluation_mode(model), torch.no_grad(): latent_tail, _ = model(tail_batch[0])
            tail_critic.train()
            for _ in range(cfg.critic_steps):
                tail_opt.zero_grad(set_to_none=True)
                tail_loss, tail_acc, _ = recipe.density_ratio_critic_loss(
                    tail_critic, latent_tail.detach(), tail_batch[1], tail_batch[2], tail_batch[4])
                tail_loss.backward(); tail_opt.step()
            tail_opt.zero_grad(set_to_none=True)
            penalty_batch = recipe._move_batch(next(stream), device)
        x, labels, nuisance, _, weights, _ = recipe._move_batch(batch, device)
        model.train(); critic.eval()
        latent, logits = model(x)
        ce = F.cross_entropy(logits, labels, reduction="none")
        with recipe.frozen_parameters(critic): information = recipe.engineer_information_penalty(critic, latent, labels, nuisance)
        ce_loss, info_loss = (ce*weights).mean(), (information*weights).mean()
        contrast = contrastive(model.get_embeddings(latent), labels, weights)
        original_loss = ce_loss + cfg.lambda_info*info_loss + cfg.contrast_weight*contrast
        optimizer.zero_grad(set_to_none=True); original_loss.backward()
        extra = torch.zeros((), device=device)
        if coefficient:
            with isolated_rng(cfg.manualSeed + epoch*100000 + step + 50000000):
                tail_critic.eval()
                with evaluation_mode(model), recipe.frozen_parameters(tail_critic):
                    latent_extra, _ = model(penalty_batch[0])
                    extra = recipe.engineer_information_penalty(tail_critic, latent_extra, penalty_batch[1], penalty_batch[2]).mean()
                (coefficient*extra).backward()
        if not torch.isfinite(original_loss.detach()+coefficient*extra.detach()): raise RuntimeError("Nonfinite training objective")
        optimizer.step()
        metrics.update_main(logits.detach(), labels, weights, ce.detach(), original_loss.detach()+coefficient*extra.detach(), info_loss, contrast)
        tail_metrics.append((float(tail_loss.detach()), float(tail_acc), float(extra.detach())))
    result = metrics.summary()
    result["tail_critic_loss"], result["tail_critic_accuracy"], result["tail_information"] = np.mean(tail_metrics, axis=0).tolist()
    return result


def train(args, recipe):
    folder = args.output_dir/args.arm
    if folder.exists(): raise FileExistsError(folder)
    report = json.loads((args.output_dir/"preparation_complete.json").read_text())
    if report["protocol"] != PROTOCOL or sha(args.output_dir/"prepared_recipe.pth.tar") != report["prepared_sha256"]:
        raise ValueError("Prepared experiment changed")
    if sha(args.output_dir/"matched_initial_state.pth.tar") != report["initial_sha256"]: raise ValueError("Initialization changed")
    prepared = torch.load(args.output_dir/"prepared_recipe.pth.tar", map_location="cpu", weights_only=False)
    cfg = argparse.Namespace(**prepared["config"])
    if args.epochs < args.tail_warmup_epochs + args.tail_ramp_epochs + args.min_post_ramp_epochs:
        raise ValueError("Insufficient complete post-ramp training budget")
    raw = torch.load(args.sample, map_location="cpu", weights_only=False)
    from utils.hlt_weights import load_generator_weights, sample_signature
    if sample_signature(raw) != prepared["data_signature"]: raise ValueError("Training data changed")
    _, metadata = load_generator_weights(str(args.weights), raw["label"], sample=raw, qcd_label=1)
    if metadata["effective_physics_weight_sha256"] != prepared["weight_metadata"]["effective_physics_weight_sha256"]:
        raise ValueError("Training weights changed")
    pre = prepared["preprocessing"]; labels = raw["label"].long().reshape(-1)
    def dataset(rows, effective):
        return make_dataset(recipe, raw["pf"], labels, prepared["scores"], prepared["physics"], pre, rows, effective)
    train_data = dataset(pre["split"]["train_indices"], prepared["train_effective"])
    cal_data = dataset(pre["split"]["validation_indices"], prepared["cal_effective"])
    device = torch.device(args.device)
    model = make_model(recipe, vars(cfg), prepared["tokens"], device)
    critic = recipe.HLTCritic(cfg.latent_dim, 4).to(device)
    initial = torch.load(args.output_dir/"matched_initial_state.pth.tar", map_location="cpu", weights_only=False)
    model.load_state_dict(initial["state_dict_model"]); critic.load_state_dict(initial["state_dict_critic"])
    if args.checkpoint_blocks: checkpoint_blocks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.critic_weight_decay)
    horizon = cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=cfg.lr*1e-3) if cfg.cosine else None
    loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type=="cuda", drop_last=True)
    cal_loader = DataLoader(cal_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)
    if not len(loader): raise ValueError("Original supervised batch larger than training population")
    tail_critic = tail_opt = stream = None
    if args.arm == "double":
        with isolated_rng(cfg.manualSeed+90000): tail_critic = recipe.HLTCritic(cfg.latent_dim, 4).to(device)
        tail_opt = torch.optim.Adam(tail_critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.critic_weight_decay)
        rows = prepared["tail_QCD_indices"]
        tail_data = dataset(rows, torch.ones(len(rows)))
        stream = TailStream(tail_data, prepared["physics"][rows], args.tail_batch_size, cfg.manualSeed+7000)
    restore_rng(initial["rng"])
    contrastive = recipe.SupConLoss(cfg.contrast_temp)
    folder.mkdir(); history = []; best = historical_best = float("inf")
    historical_bad = 0; historical_stop_epoch = None
    selection_start = args.tail_warmup_epochs + args.tail_ramp_epochs + 1
    started = time.monotonic()
    print(f"{args.arm}: FRESH seed-matched model, original batch={cfg.batch_size}, SupCon={cfg.contrast_weight}, "
          f"global lambda=1; {args.epochs} epochs, NO early stopping; selection epoch>={selection_start}", flush=True)
    for epoch in range(1, args.epochs+1):
        coefficient = tail_coefficient(epoch, args.tail_warmup_epochs, args.tail_ramp_epochs, args.tail_lambda) if tail_critic else 0.
        if tail_critic is None:
            train_metrics = recipe.train_epoch(model, critic, loader, optimizer, critic_opt, device, cfg, contrastive)
        else:
            train_metrics = train_double_epoch(recipe, model, critic, tail_critic, loader, stream, optimizer,
                critic_opt, tail_opt, device, cfg, contrastive, coefficient, epoch)
        val = recipe.validate(model, critic, cal_loader, device)
        if scheduler is not None and epoch <= horizon: scheduler.step()
        if not np.isfinite(val["weighted_ce"]): raise RuntimeError("Validation divergence; original checkpoint remains preserved")
        history.append({"epoch": epoch, "train": train_metrics, "validation": val,
                        "global_lambda": 1., "tail_lambda": coefficient, "lr": optimizer.param_groups[0]["lr"]})
        write_json(folder/"training_history.json", history)
        print(recipe.format_metrics(args.arm, epoch, val)+f" tail_lambda={coefficient:.4f} elapsed={(time.monotonic()-started)/3600:.2f}h", flush=True)
        config = {**vars(cfg), "epochs": args.epochs, "lr_schedule_epochs": horizon}
        payload = {"state_dict_model": model.state_dict(), "state_dict_critic": critic.state_dict(),
            "state_dict_tail_critic": tail_critic.state_dict() if tail_critic else None,
            "epoch": epoch, "config": config, "preprocessing": pre, "ae_scaler": prepared["ae_scaler"],
            "ae_checkpoint_sha256": prepared["ae_checkpoint_sha256"], "validation_metrics": val,
            "pilot_protocol": PROTOCOL, "arm": args.arm, "global_lambda": 1., "effective_tail_lambda": coefficient,
            "optimizer": optimizer.state_dict(), "critic_optimizer": critic_opt.state_dict(),
            "tail_optimizer": tail_opt.state_dict() if tail_opt else None,
            "source_golden_sha256": report["golden_sha256"], "actual_maximum_epochs": args.epochs,
            "selection_start_epoch": selection_start, "activation_checkpointing": args.checkpoint_blocks,
            "legacy_commit": LEGACY_COMMIT, "experiment_commit": args.code_commit}
        if historical_stop_epoch is None:
            if val["weighted_ce"] < historical_best:
                historical_best = val["weighted_ce"]; historical_bad = 0
                torch.save({**payload, "checkpoint_role": "historical_best_ce_before_original_early_stop"}, folder/"checkpoint_historical_best.pth.tar")
            else:
                historical_bad += 1
                if historical_bad >= cfg.patience: historical_stop_epoch = epoch
        if epoch >= selection_start and val["weighted_ce"] < best:
            best = val["weighted_ce"]
            torch.save({**payload, "checkpoint_role": "best_validation_ce_after_tail_ramp"}, folder/"checkpoint_main.pth.tar")
        if epoch % args.save_every == 0:
            torch.save({**payload, "checkpoint_role": "periodic_no_selection"}, folder/f"checkpoint_epoch_{epoch:03d}.pth.tar")
    torch.save({**payload, "checkpoint_role": "final_fixed_budget"}, folder/"checkpoint_final.pth.tar")
    write_json(folder/"training_complete.json", {"protocol": PROTOCOL, "arm": args.arm, "epochs_trained": args.epochs,
        "checkpoint_sha256": sha(folder/"checkpoint_main.pth.tar"), "final_sha256": sha(folder/"checkpoint_final.pth.tar"),
        "selection_start_epoch": selection_start, "post_ramp_epochs": args.epochs-selection_start+1,
        "early_stopping": False, "simulated_historical_early_stop_epoch": historical_stop_epoch,
        "historical_selection_value": historical_best,
        "source_golden_sha256": report["golden_sha256"], "initial_sha256": report["initial_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train"))
    for name in ("legacy-code", "sample", "weights", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--golden-checkpoint", type=Path)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--arm", choices=("single", "double"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--tail-warmup-epochs", type=int, default=5)
    parser.add_argument("--tail-ramp-epochs", type=int, default=10)
    parser.add_argument("--min-post-ramp-epochs", type=int, default=25)
    parser.add_argument("--tail-lambda", type=float, default=1.)
    parser.add_argument("--tail-batch-size", type=int, default=512)
    parser.add_argument("--checkpoint-blocks", type=int, choices=(0,1), default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--code-commit", default="")
    args = parser.parse_args()
    if (args.threads < 1 or args.epochs < 1 or args.min_post_ramp_epochs < 1
            or min(args.tail_warmup_epochs, args.tail_ramp_epochs) < 0 or args.tail_batch_size < 2
            or args.save_every < 1 or not np.isfinite(args.tail_lambda) or args.tail_lambda <= 0
            or not 0 < args.reproduction_tolerance <= .01): parser.error("Invalid budget, sampling or reproduction tolerance")
    if args.phase == "prepare" and (args.golden_checkpoint is None or args.ae_checkpoint is None): parser.error("Preparation requires golden model and AE")
    if args.phase == "train" and args.arm is None: parser.error("Training requires --arm")
    torch.set_num_threads(args.threads)
    recipe = pinned_recipe(args.legacy_code)
    torch.backends.cudnn.benchmark = args.device == "cuda"
    if args.phase == "prepare": prepare(args, recipe)
    else: train(args, recipe)


if __name__ == "__main__": main()
