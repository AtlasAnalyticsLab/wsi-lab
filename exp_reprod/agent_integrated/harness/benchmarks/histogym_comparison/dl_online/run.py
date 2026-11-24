#!/usr/bin/env python3
"""Run one native-observation recurrent-PPO efficiency experiment."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:  # pragma: no cover - operator convenience
    _ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_ROOT))

from benchmarks.histogym_comparison.common.resource_monitor import ResourceMonitor
from benchmarks.histogym_comparison.dl_online.backends import (
    HistoGymBackend,
    WSILabBackend,
)
from benchmarks.histogym_comparison.dl_online.common import (
    DEFAULT_PROTOCOL,
    NativeObservationActorCritic,
    NativeRolloutBuffer,
    action_counts,
    load_protocol,
    nanosecond_summary,
    observation_nbytes,
    observation_to_device,
    pack_observation,
    ppo_config,
    seed_everything,
    sha256_file,
)
from benchmarks.histogym_comparison.wsi.runner import DEFAULT_CONFIG
from wsi_lab.rl.recurrent_ppo import annealed_learning_rate, ppo_update

ROOT = Path(__file__).resolve().parents[3]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _versions() -> dict[str, Any]:
    import gymnasium
    import openslide
    import shapely

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "openslide_python": openslide.__version__,
        "shapely": shapely.__version__,
    }


def _preparation(workspace: Path | None) -> dict[str, Any]:
    if workspace is None:
        return {
            "applicable": False,
            "total_preprocessing_ns": 0,
            "reason": "HistoGym has no durable offline preprocessing stage",
        }
    source = workspace / "results/preparation.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("phase") != "preparation":
        raise ValueError(f"incomplete WSI-Lab preparation result: {source}")
    return {
        "applicable": True,
        "path": str(source),
        "sha256": sha256_file(source),
        "total_preprocessing_ns": int(value["aggregate"]["total_preprocessing_ns"]),
        "resources": value.get("resources"),
        "backend_generated_storage": value["aggregate"].get("backend_generated_storage"),
        "original_embedding_generation_included": False,
    }


def _checkpoint_snapshot(
    *,
    episode: int,
    interactions: int,
    online_started: int,
    monitor: ResourceMonitor,
) -> dict[str, Any]:
    return {
        "episode": int(episode),
        "interactions": int(interactions),
        "cumulative_online_outer_ns": int(time.perf_counter_ns() - online_started),
        "resources": monitor.prefix_rss_snapshot(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_protocol(args.protocol)
    if args.output_dir.exists():
        raise FileExistsError(f"fresh output directory must not exist: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir()
    result_path = args.output_dir / "result.json"
    failure_path = args.output_dir / "failed.json"
    trace_path = raw_dir / "trace.npz"
    checkpoint_path = args.output_dir / "final_policy.pt"

    episodes = protocol.episodes
    if args.episode_limit is not None:
        if not args.pilot:
            raise ValueError("--episode-limit is allowed only with --pilot")
        episodes = episodes[: int(args.episode_limit)]
    if not episodes:
        raise ValueError("experiment must contain at least one episode")
    interactions_per_episode = protocol.interactions_per_episode
    total_interactions = len(episodes) * interactions_per_episode
    base_ppo = ppo_config(protocol)
    if total_interactions % base_ppo.rollout_steps:
        raise ValueError(
            f"interaction count {total_interactions} must be divisible by "
            f"rollout_steps={base_ppo.rollout_steps}"
        )
    checkpoints = sorted(
        set([point for point in protocol.checkpoints if point <= len(episodes)] + [len(episodes)])
    )
    seed = int(protocol.value["policy"]["seed"])
    seed_everything(seed)
    device = torch.device(str(protocol.value["policy"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA policy device was requested but is unavailable")
    if device.type == "cuda" and device.index != int(protocol.value["measurement"]["gpu_index"]):
        raise ValueError("policy CUDA index and monitored GPU index differ")

    policy_setup_started = time.perf_counter_ns()
    model = NativeObservationActorCritic(args.backend, seed=seed)
    initial_policy_metadata = model.metadata()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_ppo.learning_rate, eps=1e-5)
    _sync(device)
    policy_setup_ns = time.perf_counter_ns() - policy_setup_started

    monitor = ResourceMonitor(
        interval_seconds=float(protocol.value["measurement"]["resource_interval_seconds"]),
        gpu_index=int(protocol.value["measurement"]["gpu_index"]),
    ).start()
    online_started = time.perf_counter_ns()
    resources: dict[str, Any] | None = None
    backend: Any | None = None
    close_ns_final = 0

    step_ns = np.zeros(total_interactions, dtype=np.int64)
    pack_ns = np.zeros(total_interactions, dtype=np.int64)
    policy_ns = np.zeros(total_interactions, dtype=np.int64)
    bootstrap_pack_ns = np.zeros(total_interactions, dtype=np.int64)
    bootstrap_value_ns = np.zeros(total_interactions, dtype=np.int64)
    bookkeeping_ns = np.zeros(total_interactions, dtype=np.int64)
    embedding_ns = np.zeros(total_interactions, dtype=np.int64)
    belief_ns = np.zeros(total_interactions, dtype=np.int64)
    actions = np.zeros(total_interactions, dtype=np.int16)
    rewards = np.zeros(total_interactions, dtype=np.float32)
    state_changed = np.zeros(total_interactions, dtype=np.bool_)
    evidence_added = np.zeros(total_interactions, dtype=np.bool_)
    observation_bytes = np.zeros(total_interactions, dtype=np.int64)
    valid_action_count = np.zeros(total_interactions, dtype=np.int16)
    episode_reset_ns = np.zeros(len(episodes), dtype=np.int64)
    episode_construction_ns = np.zeros(len(episodes), dtype=np.int64)
    episode_close_ns = np.zeros(len(episodes), dtype=np.int64)
    episode_reset_observation_bytes = np.zeros(len(episodes), dtype=np.int64)
    episode_cumulative_online_ns = np.zeros(len(episodes), dtype=np.int64)
    update_ns: list[int] = []
    update_at_interaction: list[int] = []
    update_metrics: list[dict[str, Any]] = []
    checkpoint_snapshots: list[dict[str, Any]] = []

    try:
        backend_setup_started = time.perf_counter_ns()
        if args.backend == "wsi_lab":
            if args.workspace is None:
                raise ValueError("WSI-Lab requires --workspace")
            backend = WSILabBackend(
                manifest_path=protocol.manifest_path,
                workspace=args.workspace,
                belief_checkpoint=protocol.belief_checkpoint,
                config_path=args.config,
            )
        else:
            if args.workspace is not None:
                raise ValueError("HistoGym does not accept --workspace")
            backend = HistoGymBackend(
                manifest_path=protocol.manifest_path,
                output_dir=args.output_dir,
            )
        backend_setup_ns = time.perf_counter_ns() - backend_setup_started
        if backend.num_actions != model.num_actions:
            raise AssertionError("backend/model action dimensions differ")

        buffer_allocation_started = time.perf_counter_ns()
        rollout = NativeRolloutBuffer(
            base_ppo.rollout_steps,
            backend.observation_shapes,
            backend.num_actions,
        )
        buffer_allocation_ns = time.perf_counter_ns() - buffer_allocation_started
        ppo_generator = torch.Generator(device="cpu")
        ppo_generator.manual_seed(seed + 2000)
        # Backend construction may initialize unrelated frozen modules. Reset
        # the policy sampling streams immediately before the shared workload.
        torch.manual_seed(seed + 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + 1000)

        global_index = 0
        for episode_offset, episode in enumerate(episodes):
            reset = backend.reset(episode, seed=seed)
            episode_reset_ns[episode_offset] = reset.reset_ns
            episode_construction_ns[episode_offset] = reset.construction_ns
            episode_close_ns[episode_offset] = reset.close_ns
            episode_reset_observation_bytes[episode_offset] = observation_nbytes(reset.observation)
            observation = reset.observation
            info = reset.info
            hidden = model.initial_state(1, device=device)
            episode_start = True
            for step_in_episode in range(interactions_per_episode):
                index = global_index
                pack_started = time.perf_counter_ns()
                packed = pack_observation(args.backend, observation)
                pack_ns[index] = time.perf_counter_ns() - pack_started

                mask_array = backend.action_mask(info)
                valid_action_count[index] = int(mask_array.sum())
                model.eval()
                _sync(device)
                policy_started = time.perf_counter_ns()
                device_observation = observation_to_device(packed, device)
                action_mask = torch.as_tensor(mask_array[None, :], dtype=torch.bool, device=device)
                with torch.inference_mode():
                    output = model.get_action_and_value(
                        device_observation,
                        hidden,
                        torch.as_tensor([episode_start], dtype=torch.bool, device=device),
                        action_mask,
                    )
                    action_value = int(output.action.item())
                _sync(device)
                policy_ns[index] = time.perf_counter_ns() - policy_started

                step_result = backend.step(action_value, info)
                if step_result.terminated:
                    raise RuntimeError(
                        f"{args.backend} terminated before the external boundary at "
                        f"episode {episode.episode_index}, step {step_in_episode}"
                    )
                external_truncation = step_in_episode == interactions_per_episode - 1
                truncation_value: torch.Tensor | None = None
                if external_truncation:
                    bootstrap_pack_started = time.perf_counter_ns()
                    next_packed = pack_observation(args.backend, step_result.observation)
                    bootstrap_pack_ns[index] = time.perf_counter_ns() - bootstrap_pack_started
                    _sync(device)
                    bootstrap_started = time.perf_counter_ns()
                    with torch.inference_mode():
                        truncation_value, _ = model.get_value(
                            observation_to_device(next_packed, device),
                            output.hidden_state,
                            torch.zeros(1, dtype=torch.bool, device=device),
                        )
                        _ = float(truncation_value.item())
                    _sync(device)
                    bootstrap_value_ns[index] = time.perf_counter_ns() - bootstrap_started

                bookkeeping_started = time.perf_counter_ns()
                rollout.add(
                    packed,
                    hidden,
                    episode_start,
                    action_mask,
                    output.action,
                    output.log_prob,
                    output.value,
                    step_result.reward,
                    terminated=False,
                    truncated=external_truncation,
                    truncation_value=truncation_value,
                )
                bookkeeping_ns[index] = time.perf_counter_ns() - bookkeeping_started
                step_ns[index] = step_result.step_ns
                embedding_ns[index] = step_result.embedding_lookup_ns
                belief_ns[index] = step_result.belief_inference_ns
                actions[index] = action_value
                rewards[index] = step_result.reward
                state_changed[index] = step_result.state_changed
                evidence_added[index] = step_result.evidence_added
                observation_bytes[index] = observation_nbytes(step_result.observation)
                global_index += 1

                observation = step_result.observation
                info = step_result.info
                hidden = output.hidden_state
                episode_start = False

                if rollout.full:
                    # The agreed rollout length (800) aligns with eight exact
                    # 100-step truncation boundaries, so this is always safe.
                    if not external_truncation:
                        raise AssertionError("PPO rollout ended inside a benchmark episode")
                    rollout.finish(
                        torch.zeros(1),
                        gamma=base_ppo.gamma,
                        gae_lambda=base_ppo.gae_lambda,
                    )
                    learning_rate = annealed_learning_rate(
                        base_ppo.learning_rate,
                        global_index,
                        total_interactions,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    _sync(device)
                    update_started = time.perf_counter_ns()
                    metrics = ppo_update(
                        model,
                        optimizer,
                        rollout,
                        base_ppo,
                        generator=ppo_generator,
                    )
                    _sync(device)
                    update_ns.append(time.perf_counter_ns() - update_started)
                    update_at_interaction.append(global_index)
                    update_metrics.append(
                        {
                            "interaction": global_index,
                            "learning_rate": learning_rate,
                            **metrics.to_dict(),
                        }
                    )
                    rollout.reset()
                    model.eval()

            if episode_offset == len(episodes) - 1:
                close_ns_final = int(backend.close())
                episode_close_ns[episode_offset] += close_ns_final
            episode_cumulative_online_ns[episode_offset] = time.perf_counter_ns() - online_started
            completed = episode_offset + 1
            if completed in checkpoints:
                checkpoint_snapshots.append(
                    _checkpoint_snapshot(
                        episode=completed,
                        interactions=global_index,
                        online_started=online_started,
                        monitor=monitor,
                    )
                )
                print(
                    f"{args.backend}: {completed}/{len(episodes)} episodes, "
                    f"{global_index}/{total_interactions} interactions",
                    flush=True,
                )

        if global_index != total_interactions or not rollout._cursor == 0:
            raise AssertionError("final PPO interaction/update accounting drifted")
        resources = monitor.stop()
        online_outer_ns = time.perf_counter_ns() - online_started
        backend_metadata = backend.metadata()

        # Serialization and checkpoint storage are deliberately after resource
        # monitoring and outside the online wall interval.
        serialization_started = time.perf_counter_ns()
        np.savez_compressed(
            trace_path,
            step_ns=step_ns,
            pack_ns=pack_ns,
            policy_ns=policy_ns,
            bootstrap_pack_ns=bootstrap_pack_ns,
            bootstrap_value_ns=bootstrap_value_ns,
            bookkeeping_ns=bookkeeping_ns,
            embedding_ns=embedding_ns,
            belief_ns=belief_ns,
            actions=actions,
            rewards=rewards,
            state_changed=state_changed,
            evidence_added=evidence_added,
            observation_bytes=observation_bytes,
            valid_action_count=valid_action_count,
            episode_reset_ns=episode_reset_ns,
            episode_construction_ns=episode_construction_ns,
            episode_close_ns=episode_close_ns,
            episode_reset_observation_bytes=episode_reset_observation_bytes,
            episode_cumulative_online_ns=episode_cumulative_online_ns,
            update_ns=np.asarray(update_ns, dtype=np.int64),
            update_at_interaction=np.asarray(update_at_interaction, dtype=np.int64),
        )
        torch.save(
            {
                "backend": args.backend,
                "protocol_sha256": protocol.sha256,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "interactions": total_interactions,
                "initial_policy_metadata": initial_policy_metadata,
            },
            checkpoint_path,
        )
        serialization_ns = time.perf_counter_ns() - serialization_started

        preparation = _preparation(args.workspace if args.backend == "wsi_lab" else None)
        preprocessing_ns = int(preparation["total_preprocessing_ns"])
        step_sum = int(step_ns.sum(dtype=np.int64))
        reset_sum = int(episode_reset_ns.sum(dtype=np.int64))
        construction_sum = int(episode_construction_ns.sum(dtype=np.int64))
        close_sum = int(episode_close_ns.sum(dtype=np.int64))
        pack_sum = int(pack_ns.sum(dtype=np.int64))
        bootstrap_pack_sum = int(bootstrap_pack_ns.sum(dtype=np.int64))
        policy_sum = int(policy_ns.sum(dtype=np.int64))
        bootstrap_value_sum = int(bootstrap_value_ns.sum(dtype=np.int64))
        bookkeeping_sum = int(bookkeeping_ns.sum(dtype=np.int64))
        update_sum = int(np.asarray(update_ns, dtype=np.int64).sum(dtype=np.int64))
        accounted = sum(
            (
                int(backend_setup_ns),
                int(buffer_allocation_ns),
                reset_sum,
                construction_sum,
                close_sum,
                step_sum,
                pack_sum,
                bootstrap_pack_sum,
                policy_sum,
                bootstrap_value_sum,
                bookkeeping_sum,
                update_sum,
            )
        )
        state_change_count = int(state_changed.sum())
        end_to_end_ns = preprocessing_ns + policy_setup_ns + online_outer_ns
        result = {
            "schema_version": 1,
            "status": "complete",
            "backend": args.backend,
            "pilot_excluded": bool(args.pilot),
            "scope": "computational efficiency only",
            "protocol": {
                "path": str(protocol.path),
                "sha256": protocol.sha256,
                "manifest_path": str(protocol.manifest_path),
                "manifest_sha256": sha256_file(protocol.manifest_path),
                "slide_schedule_path": str(protocol.schedule_path),
                "slide_schedule_sha256": sha256_file(protocol.schedule_path),
            },
            "runtime": {
                "versions": _versions(),
                "device": str(device),
            },
            "workload": {
                "episode_count": len(episodes),
                "unique_slide_count": len({episode.slide_id for episode in episodes}),
                "interactions_per_episode": interactions_per_episode,
                "total_interactions": total_interactions,
                "state_changing_interactions": state_change_count,
                "no_op_interactions": total_interactions - state_change_count,
                "external_truncations": len(episodes),
                "native_terminations": 0,
            },
            "policy": {
                "initial": initial_policy_metadata,
                "ppo": base_ppo.to_dict(),
                "optimizer": "Adam(eps=1e-5)",
                "policy_setup_ns": int(policy_setup_ns),
                "ppo_update_count": len(update_ns),
                "updates": update_metrics,
                "final_checkpoint": str(checkpoint_path),
                "final_checkpoint_sha256": sha256_file(checkpoint_path),
            },
            "preparation": preparation,
            "components_ns": {
                "policy_model_and_optimizer_setup": int(policy_setup_ns),
                "backend_import_and_setup": int(backend_setup_ns),
                "rollout_buffer_allocation": int(buffer_allocation_ns),
                "native_construction": construction_sum,
                "native_reset": reset_sum,
                "native_step": step_sum,
                "native_close": close_sum,
                "observation_pack": pack_sum,
                "truncation_observation_pack": bootstrap_pack_sum,
                "policy_forward_and_action": policy_sum,
                "truncation_value_forward": bootstrap_value_sum,
                "rollout_bookkeeping": bookkeeping_sum,
                "ppo_optimization": update_sum,
                "online_outer": int(online_outer_ns),
                "online_accounted_nonoverlapping": int(accounted),
                "online_other_residual": max(0, int(online_outer_ns) - int(accounted)),
                "preprocessing": preprocessing_ns,
                "end_to_end": int(end_to_end_ns),
                "embedding_lookup_nested_in_native_step": int(embedding_ns.sum(dtype=np.int64)),
                "frozen_belief_nested_in_native_step": int(belief_ns.sum(dtype=np.int64)),
            },
            "latency": {
                "native_step": nanosecond_summary(step_ns),
                "native_reset": nanosecond_summary(episode_reset_ns),
                "observation_pack": nanosecond_summary(pack_ns),
                "policy_forward_and_action": nanosecond_summary(policy_ns),
                "truncation_value_forward": nanosecond_summary(
                    bootstrap_value_ns[bootstrap_value_ns > 0]
                ),
                "rollout_bookkeeping": nanosecond_summary(bookkeeping_ns),
                "ppo_optimization": nanosecond_summary(np.asarray(update_ns, dtype=np.int64)),
                "embedding_lookup": nanosecond_summary(embedding_ns[embedding_ns > 0]),
                "frozen_belief": nanosecond_summary(belief_ns[belief_ns > 0]),
            },
            "throughput": {
                "native_step_calls_per_second": (
                    total_interactions / (step_sum / 1e9) if step_sum else None
                ),
                "state_changes_per_second_native_step": (
                    state_change_count / (step_sum / 1e9) if step_sum else None
                ),
                "overall_online_interactions_per_second": (
                    total_interactions / (online_outer_ns / 1e9)
                ),
                "end_to_end_interactions_per_second": (total_interactions / (end_to_end_ns / 1e9)),
                "online_seconds_per_episode": online_outer_ns / 1e9 / len(episodes),
                "end_to_end_seconds_per_episode": end_to_end_ns / 1e9 / len(episodes),
            },
            "observations": {
                "reset_plus_post_step_payload_bytes": int(
                    episode_reset_observation_bytes.sum(dtype=np.int64)
                    + observation_bytes.sum(dtype=np.int64)
                ),
                "post_step_payload_bytes": int(observation_bytes.sum(dtype=np.int64)),
            },
            "actions": action_counts(actions, backend.num_actions),
            "belief": backend_metadata,
            "resources": resources,
            "checkpoints": checkpoint_snapshots,
            "artifacts": {
                "trace": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
                "trace_serialization_and_checkpoint_ns_excluded": int(serialization_ns),
                "checkpoint_logical_bytes": checkpoint_path.stat().st_size,
                "trace_logical_bytes": trace_path.stat().st_size,
            },
        }
        atomic_json(result_path, result)
        return result
    except BaseException as exc:
        if resources is None:
            try:
                resources = monitor.stop()
            except BaseException:
                resources = None
        if backend is not None:
            with contextlib.suppress(BaseException):
                backend.close()
        failure = {
            "schema_version": 1,
            "status": "failed",
            "backend": args.backend,
            "pilot_excluded": bool(args.pilot),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "resources": resources,
        }
        atomic_json(failure_path, failure)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("wsi_lab", "histogym"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--episode-limit", type=int)
    args = parser.parse_args(argv)
    args.protocol = args.protocol.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.workspace = args.workspace.expanduser().resolve() if args.workspace else None
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.episode_limit is not None and args.episode_limit <= 0:
        parser.error("--episode-limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "backend": result["backend"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
