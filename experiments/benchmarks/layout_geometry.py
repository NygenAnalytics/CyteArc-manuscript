import math

import numpy as np
from cytearc.storage.sharding import DEFAULT_COUNT_MATRIX_POLICY, plan_count_matrix_pair

from experiments.benchmarks import HERE, RAW
from experiments.runtime.common import BENCHMARK_GENES, BENCHMARK_SIZES, write_csv, write_json
from experiments.runtime.records import load_runs


def stored_dtypes() -> dict[int, str]:
    found: dict[int, set[str]] = {}
    for run in load_runs("rna_scaling_runs", RAW).values():
        funnel = run.get("funnel")
        if funnel is None:
            continue
        outcomes = {o["stage"]: o for o in funnel["outcomes"]}
        dtype = (outcomes.get("writeCountsT", {}).get("details") or {}).get("dtype")
        if dtype:
            found.setdefault(int(funnel["nRows"]), set()).add(str(dtype))
    for n_cells, dtypes in found.items():
        if len(dtypes) != 1:
            raise SystemExit(f"{n_cells:,} cells: the replicates stored {sorted(dtypes)}")
    return {n: next(iter(d)) for n, d in found.items()}


def main() -> None:
    policy = DEFAULT_COUNT_MATRIX_POLICY
    dtypes = stored_dtypes()
    rows = []
    for n_cells in BENCHMARK_SIZES:
        dtype = np.dtype(dtypes[n_cells])
        element_bytes = dtype.itemsize
        plan = plan_count_matrix_pair(n_cells, BENCHMARK_GENES, dtype, profile="cloud")
        counts_chunks = tuple((int(v) for v in plan.counts.chunks))
        counts_shards = tuple((int(v) for v in plan.counts.shards or plan.counts.chunks))
        counts_t_chunks = tuple((int(v) for v in plan.countsT.chunks))
        counts_t_shards = tuple((int(v) for v in plan.countsT.shards or plan.countsT.chunks))
        group = plan.readGroup
        chunk_gene_width = counts_chunks[1]
        read_width = int(group.featureWidth)
        rows.append(
            {
                "nCells": n_cells,
                "nGenes": BENCHMARK_GENES,
                "dtype": dtype.name,
                "elementBytes": element_bytes,
                "unitBytes": policy.unitBytes,
                "chunkBytes": policy.chunkBytes,
                "countsChunkCells": counts_chunks[0],
                "countsChunkGenes": counts_chunks[1],
                "countsShardCells": counts_shards[0],
                "countsShardGenes": counts_shards[1],
                "countsUnits": math.ceil(n_cells / counts_shards[0]),
                "countsTChunkGenes": counts_t_chunks[0],
                "countsTChunkCells": counts_t_chunks[1],
                "countsTShardGenes": counts_t_shards[0],
                "countsTShardCells": counts_t_shards[1],
                "geneReadWidth": read_width,
                "geneReadUnits": math.ceil(BENCHMARK_GENES / read_width),
                "readGroupBytes": int(group.readGroupBytes),
                "readGroupChunksTouched": int(group.chunksTouched),
                "readGroupShardsTouched": int(group.shardsTouched),
                "amplificationCellMajorOnly": round(chunk_gene_width / read_width, 2),
                "amplificationFormula": round(
                    chunk_gene_width * n_cells * element_bytes / policy.unitBytes, 2
                ),
                "decodeAmplificationPairedWrite": float(plan.sourceDecodeAmplification),
                "denseLogicalBytes": n_cells * BENCHMARK_GENES * element_bytes,
                "layoutFingerprint": plan.fingerprint,
            }
        )
    write_csv("count_layout_geometry.csv", rows, directory=HERE / "records")
    write_json(
        "count_layout_geometry.json",
        {
            "description": "Paired counts/countsT geometry from plan_count_matrix_pair for the benchmark dataset, with the value type that each size's import stored. amplificationCellMajorOnly is w_c / w_r, the bytes decoded per byte used by a gene-wise pass on the cell-major layout; the paired layout makes it 1.",
            "policy": {"unitBytes": policy.unitBytes, "chunkBytes": policy.chunkBytes},
            "rows": rows,
        },
        directory=HERE / "records",
    )


if __name__ == "__main__":
    main()
