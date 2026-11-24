#!/usr/bin/env python3
"""Create the one immutable cohort manifest used by both backends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


EMBEDDING_VARIANTS = (
    "UNI_v2_2.5x_256",
    "UNI_v2_5x_256",
    "UNI_v2_10x_256",
    "UNI_v2_20x_256",
)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def stat_record(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
        "mtime_ns": stat.st_mtime_ns,
    }


def xml_counts(path: Path) -> dict[str, int]:
    root = ElementTree.parse(path).getroot()
    return {
        "annotations": sum(1 for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Annotation"),
        "coordinates": sum(1 for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Coordinate"),
    }


def build_row(slide_id: str, dataset_root: Path, ordinal: int) -> dict:
    wsi_path = dataset_root / "images" / f"{slide_id}.tif"
    xml_path = dataset_root / "annotations" / f"{slide_id}.xml"
    embeddings = {}
    for variant in EMBEDDING_VARIANTS:
        embedding_path = dataset_root / "Embeddings" / variant / "patches" / f"{slide_id}.h5"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Missing required existing embedding: {embedding_path}")
        embeddings[variant] = stat_record(embedding_path)

    return {
        "ordinal": ordinal,
        "slide_id": slide_id,
        "label": 1,
        "wsi_path": str(wsi_path.resolve()),
        "xml_path": str(xml_path.resolve()),
        "metadata": {
            "wsi": stat_record(wsi_path),
            "xml": {
                **stat_record(xml_path),
                "sha256": sha256_file(xml_path),
                **xml_counts(xml_path),
            },
            "embeddings": embeddings,
        },
    }


def atomic_json_dump(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "manifests" / "cohort.json",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--measured-size", type=int, default=100)
    parser.add_argument("--expected-population-size", type=int, default=111)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    images = dataset_root / "images"
    annotations = dataset_root / "annotations"

    reference_path = dataset_root / "evaluation" / "reference.csv"
    population_ids = []
    with reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            slide_id = Path((row.get("image") or "").strip()).stem
            label_name = (row.get("type") or "").strip().casefold()
            if label_name == "tumor" and not slide_id.startswith("test_"):
                population_ids.append(slide_id)
    population = sorted(population_ids)
    if len(population) != len(set(population)):
        raise RuntimeError("The authoritative CAMELYON16 reference contains duplicate slide IDs")
    missing_inputs = [
        slide_id
        for slide_id in population
        if not (images / f"{slide_id}.tif").is_file()
        or not (annotations / f"{slide_id}.xml").is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"Authoritative cohort is missing WSI/XML inputs: {missing_inputs[:8]}"
        )
    if len(population) != args.expected_population_size:
        raise RuntimeError(
            f"Expected {args.expected_population_size} annotated tumor slides; "
            f"found {len(population)}"
        )
    if not 0 < args.measured_size < len(population):
        raise ValueError("measured-size must leave at least one pilot slide")

    shuffled = population.copy()
    random.Random(args.seed).shuffle(shuffled)
    measured_ids = shuffled[: args.measured_size]
    pilot_ids = shuffled[args.measured_size :]

    population_digest = hashlib.sha256("\n".join(population).encode()).hexdigest()
    measured_digest = hashlib.sha256("\n".join(measured_ids).encode()).hexdigest()
    pilot_digest = hashlib.sha256("\n".join(pilot_ids).encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "population_definition": "authoritative evaluation/reference.csv rows with type=tumor and official development (non-test) slide IDs, requiring XML",
        "reference_table": {
            **stat_record(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "sampling": {
            "seed": args.seed,
            "algorithm": "sorted population then Python random.Random(seed).shuffle (MT19937)",
            "population_size": len(population),
            "measured_size": len(measured_ids),
            "pilot_size": len(pilot_ids),
            "population_ids_sha256": population_digest,
            "measured_ids_sha256": measured_digest,
            "pilot_ids_sha256": pilot_digest,
        },
        "slides": [
            build_row(slide_id, dataset_root, ordinal)
            for ordinal, slide_id in enumerate(measured_ids)
        ],
        "pilot_slides": [
            build_row(slide_id, dataset_root, ordinal)
            for ordinal, slide_id in enumerate(pilot_ids)
        ],
    }
    atomic_json_dump(payload, args.output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "measured": len(measured_ids),
                "pilot": len(pilot_ids),
                "measured_ids_sha256": measured_digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
