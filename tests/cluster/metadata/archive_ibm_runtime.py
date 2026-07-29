from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qiskit_ibm_runtime import QiskitRuntimeService


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CSV_FILES = [
    "tests/cluster/metadata/session_3.csv",
    "tests/cluster/metadata/session_5.csv",
]

OUTPUT_FOLDER = Path("tests/cluster/metadata/archive")

# Change this after printing the CSV columns if necessary.
JOB_ID_COLUMN = "JobId"


# ------------------------------------------------------------
# Convert Qiskit and NumPy objects into JSON-compatible values
# ------------------------------------------------------------

def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, complex):
        return {
            "real": float(value.real),
            "imag": float(value.imag),
        }

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    if hasattr(value, "to_dict"):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                key: to_jsonable(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return str(value)


def get_attribute(obj: Any, name: str, default=None):
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:
        return default


def get_backend_name(job) -> str | None:
    backend = get_attribute(job, "backend")

    if backend is None:
        return None

    name = get_attribute(backend, "name")

    if name is not None:
        return str(name)

    return str(backend)


# ------------------------------------------------------------
# Read all job IDs from the downloaded CSV files
# ------------------------------------------------------------

def read_job_ids(csv_files: list[str]) -> list[str]:
    job_ids: set[str] = set()

    for csv_file in csv_files:
        dataframe = pd.read_csv(csv_file, dtype=str)

        print(f"\nColumns in {csv_file}:")
        print(list(dataframe.columns))

        if JOB_ID_COLUMN not in dataframe.columns:
            raise KeyError(
                f"Could not find column {JOB_ID_COLUMN!r} in {csv_file}.\n"
                f"Available columns: {list(dataframe.columns)}"
            )

        ids = dataframe[JOB_ID_COLUMN].dropna().astype(str).str.strip()
        job_ids.update(job_id for job_id in ids if job_id)

    return sorted(job_ids)


# ------------------------------------------------------------
# Extract useful primitive-result data
# ------------------------------------------------------------

def extract_pub_data(pub_result) -> dict[str, Any]:
    pub_data: dict[str, Any] = {
        "metadata": to_jsonable(
            get_attribute(pub_result, "metadata", {})
        ),
        "data": {},
    }

    data = get_attribute(pub_result, "data")

    if data is None:
        return pub_data

    # Common EstimatorV2 output fields.
    for field_name in ["evs", "stds", "ensemble_standard_error"]:
        if hasattr(data, field_name):
            pub_data["data"][field_name] = to_jsonable(
                getattr(data, field_name)
            )

    # Common SamplerV2 data containers and any other public fields.
    if hasattr(data, "__dict__"):
        for field_name, value in vars(data).items():
            if field_name.startswith("_"):
                continue

            if field_name not in pub_data["data"]:
                pub_data["data"][field_name] = to_jsonable(value)

    # Some DataBin objects expose keys but not useful __dict__ fields.
    if hasattr(data, "keys"):
        try:
            for field_name in data.keys():
                if field_name in pub_data["data"]:
                    continue

                try:
                    value = getattr(data, field_name)
                except Exception:
                    try:
                        value = data[field_name]
                    except Exception:
                        continue

                pub_data["data"][str(field_name)] = to_jsonable(value)

        except Exception:
            pass

    return pub_data


# ------------------------------------------------------------
# Archive one IBM job
# ------------------------------------------------------------

def archive_job(service, job_id: str, output_folder: Path) -> dict[str, Any]:
    print(f"Downloading {job_id}...")

    job_folder = output_folder / "jobs"
    job_folder.mkdir(parents=True, exist_ok=True)

    output_path = job_folder / f"{job_id}.json"

    try:
        job = service.job(job_id)

        status = str(get_attribute(job, "status", "unknown"))

        job_record: dict[str, Any] = {
            "job_id": job_id,
            "status": status,
            "session_id": get_attribute(job, "session_id"),
            "backend": get_backend_name(job),
            "primitive_id": get_attribute(job, "primitive_id"),
            "creation_date": str(
                get_attribute(job, "creation_date", "")
            ),
            "instance": get_attribute(job, "instance"),
            "tags": to_jsonable(get_attribute(job, "tags")),
            "runtime_image": get_attribute(job, "image"),
            "usage_estimation": to_jsonable(
                get_attribute(job, "usage_estimation")
            ),
            "metrics": None,
            "inputs": None,
            "logs": None,
            "result_metadata": None,
            "pub_results": [],
            "errors": {},
        }

        try:
            job_record["metrics"] = to_jsonable(job.metrics())
        except Exception as exc:
            job_record["errors"]["metrics"] = str(exc)

        try:
            job_record["inputs"] = to_jsonable(
                get_attribute(job, "inputs")
            )
        except Exception as exc:
            job_record["errors"]["inputs"] = str(exc)

        try:
            job_record["logs"] = str(job.logs())
        except Exception as exc:
            job_record["errors"]["logs"] = str(exc)

        try:
            result = job.result()

            job_record["result_metadata"] = to_jsonable(
                get_attribute(result, "metadata", {})
            )

            for pub_index, pub_result in enumerate(result):
                pub_record = extract_pub_data(pub_result)
                pub_record["pub_index"] = pub_index
                job_record["pub_results"].append(pub_record)

        except Exception as exc:
            job_record["errors"]["result"] = str(exc)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                job_record,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved {output_path}")

        return {
            "job_id": job_id,
            "session_id": job_record["session_id"],
            "backend": job_record["backend"],
            "primitive_id": job_record["primitive_id"],
            "status": job_record["status"],
            "creation_date": job_record["creation_date"],
            "json_file": str(output_path),
            "error": job_record["errors"].get("result"),
        }

    except Exception as exc:
        error_record = {
            "job_id": job_id,
            "retrieval_error": str(exc),
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(error_record, file, indent=2)

        print(f"Failed to retrieve {job_id}: {exc}")

        return {
            "job_id": job_id,
            "session_id": None,
            "backend": None,
            "primitive_id": None,
            "status": "retrieval_error",
            "creation_date": None,
            "json_file": str(output_path),
            "error": str(exc),
        }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    service = QiskitRuntimeService()

    job_ids = read_job_ids(CSV_FILES)

    print(f"\nFound {len(job_ids)} unique jobs.")

    summary_rows = []

    for index, job_id in enumerate(job_ids, start=1):
        print(f"\nJob {index}/{len(job_ids)}")

        summary = archive_job(
            service,
            job_id,
            OUTPUT_FOLDER,
        )

        summary_rows.append(summary)

    summary_dataframe = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_FOLDER / "jobs_summary.csv"
    summary_dataframe.to_csv(summary_path, index=False)

    print("\nArchive finished.")
    print("Summary:", summary_path.resolve())
    print("Job JSON files:", (OUTPUT_FOLDER / "jobs").resolve())


if __name__ == "__main__":
    main()