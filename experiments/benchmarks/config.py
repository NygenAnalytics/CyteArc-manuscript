import json
import math
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.benchmarks.datasets import DEFAULT_TARGET_SIZES
from experiments.runtime.locations import ExperimentBucket, experiment_bucket

MODAL_APP_NAME = "cytearc-experiments"
MODAL_ENVIRONMENT_NAME = "cytearc_profiling"


StageName = Literal[
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
    "runUmap",
    "runLeiden",
    "findMarkers",
    "importClusters",
    "validateExperiment",
    "makeBulkMean",
    "makeBulkSum",
    "runDoublets",
    "runDoubletsRatio01",
]
GRAPH_CONSTRUCTION_STAGE_ORDER: tuple[StageName, ...] = (
    "runNormalization",
    "runPca",
    "buildEmbeddingInitialization",
    "buildAnnIndex",
    "queryNeighbors",
    "buildConnectivityMap",
)
CORE_STAGE_ORDER: tuple[StageName, ...] = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    *GRAPH_CONSTRUCTION_STAGE_ORDER,
    "runUmap",
    "runLeiden",
    "findMarkers",
)
SELECTED_STAGE_ORDER: tuple[StageName, ...] = (
    "createStore",
    "writeCountsT",
    "initializeStore",
    "reopenStore",
    "filterCells",
    "markHvgs",
    "runNormalization",
    "runPca",
    "importClusters",
    "findMarkers",
    "validateExperiment",
)
SELECTED_STAGE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    "createStore": (),
    "writeCountsT": ("createStore",),
    "initializeStore": ("writeCountsT",),
    "reopenStore": ("initializeStore",),
    "filterCells": ("reopenStore",),
    "markHvgs": ("filterCells",),
    "runNormalization": ("markHvgs",),
    "runPca": ("runNormalization",),
    "importClusters": ("filterCells",),
    "findMarkers": ("importClusters",),
    "validateExperiment": ("runPca", "findMarkers"),
}
CONSUME_STAGE_ORDER: tuple[StageName, ...] = (
    "makeBulkMean",
    "makeBulkSum",
    "runDoublets",
    "runDoubletsRatio01",
)
CONSUME_STAGES = frozenset(CONSUME_STAGE_ORDER)
ALL_STAGE_CHOICES: tuple[StageName, ...] = tuple(
    dict.fromkeys((*CORE_STAGE_ORDER, *SELECTED_STAGE_ORDER, *CONSUME_STAGE_ORDER))
)


def validate_requested_stages(stages: tuple[StageName, ...]) -> None:
    if not stages:
        raise ValueError("stages must not be empty")
    if len(set(stages)) != len(stages):
        raise ValueError("stages must be unique")
    stages = tuple((stage for stage in stages if stage not in CONSUME_STAGES))
    if not stages:
        return
    selected = set(stages)
    positions = {stage: index for index, stage in enumerate(stages)}
    core_set = set(CORE_STAGE_ORDER)
    if selected <= core_set:
        core_index = {stage: index for index, stage in enumerate(CORE_STAGE_ORDER)}
        ordered = tuple(sorted(stages, key=lambda stage: core_index[stage]))
        if stages != ordered:
            raise ValueError("CORE stages must appear in CORE_STAGE_ORDER")
        return
    for stage in stages:
        deps = SELECTED_STAGE_DEPENDENCIES.get(stage)
        if deps is None:
            raise ValueError(f"{stage} is not in the selected-stage graph")
        for dep in deps:
            if dep not in selected:
                raise ValueError(f"{stage} requires {dep}")
            if positions[dep] >= positions[stage]:
                raise ValueError(f"{dep} must precede {stage}")


MAX_TIMEOUT_SECONDS = 86400
CLUSTER_SOURCES_PATH = Path(__file__).resolve().parent / "cluster_sources.toml"
# The settings the benchmark configurations start from.
TEMPLATE = Path(__file__).resolve().parent / "configs" / "config.toml"
# A configuration names objects by their keys in the experiment bucket; the loader turns each
# key into the URI that the model holds, with the bucket and its endpoint from the environment.
BUCKET_KEYS = {"datasetPrefix": "datasetPrefixUri", "resultsPrefix": "resultsUri"}
_ARTIFACT_ID_PATTERN = re.compile("^[0-9a-f]{64}$")


class WorkflowParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    assayName: str = "RNA"
    cellKey: str = "I"
    filterAttrs: tuple[str, ...] = (
        "RNA_nCounts",
        "RNA_nFeatures",
        "RNA_percentMito",
        "RNA_percentRibo",
    )
    filterMinQuantile: float = 0.01
    filterMaxQuantile: float = 0.99
    minFeaturesPerCell: int = 10
    h5adBatchSize: int = 1000
    topN: int = 1000
    hvgMinCells: int = 20
    k: int = 11
    dims: int = 21
    nCentroids: int = 1000
    graphSeed: int = 4466
    kmeansSampling: float = 0.1
    kmeansBatchSize: int = 10000
    annParallel: bool = False
    umapEpochs: int = 300
    umapSeed: int = 4444
    umapParallel: bool = True
    leidenResolution: float = 1.0
    leidenBackend: Literal["cytearc", "igraph", "leidenalg"] = "cytearc"
    leidenSeed: int = 4444
    graphLocalCache: bool | str = "auto"
    parisNClusters: int | Literal["auto"] = "auto"
    parisMinClusterSize: int | None = None
    clusterSourceUri: str | None = None
    clusterSourceArtifactId: str | None = None

    @model_validator(mode="after")
    def _check_workflow(self) -> Self:
        if self.cellKey != "I":
            raise ValueError("cellKey must be 'I'; profiling does not use cellKey")
        if self.parisNClusters != "auto" or self.parisMinClusterSize is not None:
            raise ValueError(
                "profiling does not run Paris; leave parisNClusters='auto' and parisMinClusterSize unset"
            )
        if not math.isfinite(self.kmeansSampling) or not 0 < self.kmeansSampling <= 1:
            raise ValueError("kmeansSampling must be greater than 0 and at most 1")
        if self.kmeansBatchSize <= 0:
            raise ValueError("kmeansBatchSize must be positive")
        if (self.clusterSourceUri is None) != (self.clusterSourceArtifactId is None):
            raise ValueError("clusterSourceUri and clusterSourceArtifactId must be set together")
        if self.clusterSourceUri is not None:
            uri = self.clusterSourceUri.strip()
            if not uri:
                raise ValueError("clusterSourceUri must be non-empty when set")
            if not (uri.startswith("s3://") or uri.startswith("/") or uri.startswith("file://")):
                raise ValueError("clusterSourceUri must be an s3:// URI or local path")
            assert self.clusterSourceArtifactId is not None
            if _ARTIFACT_ID_PATTERN.fullmatch(self.clusterSourceArtifactId) is None:
                raise ValueError(
                    "clusterSourceArtifactId must be a 64-character lowercase hex token"
                )
        return self


class StageResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modalMemoryRequestMb: int
    modalMemoryLimitMb: int
    modalCpuRequest: float
    modalCpuLimit: float
    cytearcMemoryBudget: int
    workers: int
    timeoutSeconds: int
    ephemeralDiskMb: int

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.modalMemoryRequestMb <= 0 or self.modalMemoryLimitMb <= 0:
            raise ValueError("Modal memory values must be positive")
        if self.modalMemoryLimitMb < self.modalMemoryRequestMb:
            raise ValueError("modalMemoryLimitMb must be >= modalMemoryRequestMb")
        if self.modalCpuRequest <= 0 or self.modalCpuLimit <= 0:
            raise ValueError("Modal CPU values must be positive")
        if self.modalCpuLimit < self.modalCpuRequest:
            raise ValueError("modalCpuLimit must be >= modalCpuRequest")
        if self.cytearcMemoryBudget <= 0:
            raise ValueError("cytearcMemoryBudget must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if not 0 < self.timeoutSeconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeoutSeconds must be in 1..{MAX_TIMEOUT_SECONDS}")
        if self.ephemeralDiskMb <= 0:
            raise ValueError("ephemeralDiskMb must be positive")
        return self


class PrepareResources(BaseModel):
    """Modal sizing for dataset prepare (full CSR resident in memory)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    modalMemoryRequestMb: int = 196608
    modalMemoryLimitMb: int = 212992
    modalCpuRequest: float = 8.0
    modalCpuLimit: float = 16.0
    timeoutSeconds: int = 86400
    ephemeralDiskMb: int = 524288

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.modalMemoryRequestMb <= 0 or self.modalMemoryLimitMb <= 0:
            raise ValueError("Modal memory values must be positive")
        if self.modalMemoryLimitMb < self.modalMemoryRequestMb:
            raise ValueError("modalMemoryLimitMb must be >= modalMemoryRequestMb")
        if self.modalCpuRequest <= 0 or self.modalCpuLimit <= 0:
            raise ValueError("Modal CPU values must be positive")
        if self.modalCpuLimit < self.modalCpuRequest:
            raise ValueError("modalCpuLimit must be >= modalCpuRequest")
        if not 0 < self.timeoutSeconds <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"timeoutSeconds must be in 1..{MAX_TIMEOUT_SECONDS}")
        if self.ephemeralDiskMb <= 0:
            raise ValueError("ephemeralDiskMb must be positive")
        return self


class CountMatrixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unitBytes: int = 1000000000
    chunkBytes: int = 100000000

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if min(self.unitBytes, self.chunkBytes) < 1:
            raise ValueError("countMatrix values must be positive")
        if self.unitBytes < self.chunkBytes:
            raise ValueError("unitBytes must be at least chunkBytes")
        return self


class StorageIoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    readWorkers: int | None = None
    computeWorkers: int | None = None
    writeWorkers: int | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        for name in ("readWorkers", "computeWorkers", "writeWorkers"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when set")
        return self


class ClusterSourceRef(BaseModel):
    """Cluster labels to import: a store and artifact, or another run's Leiden stage record.

    ``leidenRecordUri`` names the runLeiden record of a run on the same cells. Its store and
    artifact are read when a step that uses the imported labels starts, so the steps before it
    can run before the source run has clustered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    nRows: int
    storeUri: str | None = None
    artifactId: str | None = None
    leidenRecordUri: str | None = None

    @model_validator(mode="after")
    def _check_ref(self) -> Self:
        if self.nRows <= 0:
            raise ValueError("cluster source nRows must be positive")
        if self.leidenRecordUri is not None:
            if self.storeUri is not None or self.artifactId is not None:
                raise ValueError(
                    "Give a cluster source as leidenRecordUri or as storeUri and artifactId"
                )
            _require_store_uri(self.leidenRecordUri, "cluster source leidenRecordUri")
            return self
        if self.storeUri is None or not self.storeUri.startswith("s3://"):
            raise ValueError("cluster source storeUri must be an s3:// URI")
        if self.artifactId is None or _ARTIFACT_ID_PATTERN.fullmatch(self.artifactId) is None:
            raise ValueError("cluster source artifactId must be a 64-character lowercase hex token")
        return self

    def resolved(self) -> tuple[str, str]:
        """Return the source store and its cluster-label artifact ID."""
        if self.leidenRecordUri is None:
            assert self.storeUri is not None and self.artifactId is not None
            return self.storeUri, self.artifactId
        from experiments.runtime.object_io import get_json

        uri = self.leidenRecordUri
        record = (
            get_json(uri)
            if uri.startswith("s3://")
            else json.loads(Path(uri.removeprefix("file://")).read_text())
        )
        artifact = (record.get("details") or {}).get("artifact") or {}
        if record.get("stage") != "runLeiden" or record.get("status") != "ok":
            raise ValueError(f"{uri} is not a completed runLeiden record")
        if record.get("nRows") != self.nRows or artifact.get("kind") != "cluster_labels":
            raise ValueError(f"{uri} does not hold cluster labels for {self.nRows} cells")
        return str(record["storeUri"]), str(artifact["artifact_id"])


class ObjectReadConfig(BaseModel):
    """Steps whose S3 GETs are counted after Zarr range coalescing, as in count_requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stages: tuple[StageName, ...]
    maximumAttempts: int = 4

    @model_validator(mode="after")
    def _check_reads(self) -> Self:
        if not self.stages or len(set(self.stages)) != len(self.stages):
            raise ValueError("objectReads.stages must name distinct stages")
        if self.maximumAttempts < 1:
            raise ValueError("objectReads.maximumAttempts must be at least 1")
        return self


class TahoeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    revision: str
    fileCount: int | None = None
    prefetchFiles: int = Field(gt=0)
    maxDownloadBytes: int = Field(gt=0)


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    modalEnvironmentName: str = MODAL_ENVIRONMENT_NAME
    modalAppName: str = MODAL_APP_NAME
    modalSecretName: str
    r2EndpointUrl: str
    datasetPrefixUri: str
    dataset: Literal["embryo", "mouse_brain", "tahoe"] = "embryo"
    tahoe: TahoeInput | None = None
    inputUri: str | None = None
    resultsUri: str
    storeBackend: Literal["objectStore", "modalLocalDisk"] = "objectStore"
    runTag: str = ""
    storeUriOverride: str | None = None
    storeUriBySize: dict[int, str] = Field(default_factory=dict)
    countMatrix: CountMatrixConfig | None = None
    storageIo: StorageIoConfig | None = None
    clusterSources: tuple[ClusterSourceRef, ...] = ()
    objectReads: ObjectReadConfig | None = None
    targetSizes: tuple[int, ...] = Field(default_factory=lambda: DEFAULT_TARGET_SIZES)
    samplingSeed: int = 0
    workflow: WorkflowParameters = Field(default_factory=WorkflowParameters)
    prepareResources: PrepareResources = Field(default_factory=PrepareResources)
    stageResources: dict[StageName, StageResources]
    stages: tuple[StageName, ...] | None = None

    @property
    def effectiveStages(self) -> tuple[StageName, ...]:
        return self.stages if self.stages is not None else CORE_STAGE_ORDER

    @model_validator(mode="before")
    @classmethod
    def _coerce_store_uri_by_size(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = data.get("storeUriBySize")
        if not isinstance(raw, dict):
            return data
        payload = dict(data)
        payload["storeUriBySize"] = {int(key): value for key, value in raw.items()}
        return payload

    @model_validator(mode="after")
    def _check_config(self) -> Self:
        if self.modalEnvironmentName != MODAL_ENVIRONMENT_NAME:
            raise ValueError(f"modalEnvironmentName must be {MODAL_ENVIRONMENT_NAME}")
        if self.modalAppName != MODAL_APP_NAME:
            raise ValueError(f"modalAppName must be {MODAL_APP_NAME}")
        _require_store_uri(self.datasetPrefixUri, "datasetPrefixUri")
        if self.inputUri is not None:
            parsed = urlsplit(self.inputUri)
            if (
                parsed.scheme not in {"s3", "file", ""}
                or (parsed.scheme == "s3" and not parsed.netloc)
                or (parsed.scheme != "s3" and not parsed.path.startswith("/"))
                or (not parsed.path.strip("/"))
                or parsed.path.endswith("/")
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("inputUri must name one S3 object or absolute local H5AD file")
            if len(self.targetSizes) != 1:
                raise ValueError("inputUri requires exactly one targetSizes entry")
        _require_store_uri(self.resultsUri, "resultsUri")
        if self.dataset == "mouse_brain" and self.targetSizes != (1306127,):
            raise ValueError("mouse_brain requires targetSizes = [1306127]")
        if (self.dataset == "tahoe") != (self.tahoe is not None):
            raise ValueError("The tahoe dataset requires its input settings, and only that dataset")
        if self.tahoe is not None:
            from experiments.benchmarks.tahoe import require_revision, selected_data_files

            require_revision(self.tahoe.revision)
            selected_data_files(self.tahoe.fileCount)
        if self.storeUriOverride is not None:
            _require_store_uri(self.storeUriOverride, "storeUriOverride")
        for n_rows, uri in self.storeUriBySize.items():
            if n_rows <= 0:
                raise ValueError("storeUriBySize keys must be positive")
            _require_store_uri(uri, f"storeUriBySize[{n_rows}]")
        if "/" in self.runTag or "\\" in self.runTag or self.runTag in {".", ".."}:
            raise ValueError("runTag must be a single path segment")
        if not self.targetSizes:
            raise ValueError("targetSizes must not be empty")
        if any((size <= 0 for size in self.targetSizes)):
            raise ValueError("targetSizes must be positive")
        if tuple(sorted(self.targetSizes)) != self.targetSizes:
            raise ValueError("targetSizes must be strictly increasing")
        if len(set(self.targetSizes)) != len(self.targetSizes):
            raise ValueError("targetSizes must be unique")
        selected = self.effectiveStages
        if not selected:
            raise ValueError("stages must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("stages must be unique")
        missing = [stage for stage in selected if stage not in self.stageResources]
        if missing:
            raise ValueError(f"Missing stageResources for: {', '.join(missing)}")
        validate_requested_stages(selected)
        if self.objectReads is not None:
            if self.storeBackend != "objectStore":
                raise ValueError("objectReads counts S3 GETs, so it needs an objectStore store")
            unselected = sorted(set(self.objectReads.stages) - set(selected))
            if unselected:
                raise ValueError(f"objectReads names unselected stages: {', '.join(unselected)}")
        cluster_sizes = [item.nRows for item in self.clusterSources]
        if len(set(cluster_sizes)) != len(cluster_sizes):
            raise ValueError("clusterSources nRows must be unique")
        return self

    def datasetUri(self, nRows: int) -> str:
        if self.inputUri is not None:
            if nRows not in self.targetSizes:
                raise ValueError(f"size {nRows} is not in config.targetSizes")
            return self.inputUri
        return f"{self.datasetPrefixUri.rstrip('/')}/{nRows}.h5ad"

    def sourceUri(self) -> str:
        return f"{self.datasetPrefixUri.rstrip('/')}/source.h5ad"

    def _tagged_prefix(self, kind: str) -> str:
        base = f"{self.resultsUri.rstrip('/')}/{kind}"
        if self.runTag:
            return f"{base}/{self.runTag}"
        return base

    def storeOverrideFor(self, nRows: int) -> str | None:
        sized = self.storeUriBySize.get(nRows)
        if sized is not None:
            return sized.rstrip("/")
        if self.storeUriOverride is not None:
            return self.storeUriOverride.rstrip("/")
        return None

    def storeUri(self, nRows: int) -> str:
        override = self.storeOverrideFor(nRows)
        if override is not None:
            return override
        if self.storeBackend == "modalLocalDisk":
            return f"/tmp/rna_scaling/{self.runTag}/{nRows}.zarr"
        return f"{self._tagged_prefix('stores')}/{nRows}.zarr"

    def resultPrefix(self, nRows: int) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}"

    def resultUri(self, nRows: int, stage: StageName) -> str:
        return f"{self.resultPrefix(nRows)}/{stage}.json"

    def workflowResultUri(self, nRows: int) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/funnel.json"

    def workflowClaimUri(self) -> str:
        return f"{self._tagged_prefix('results')}/e2e-claim.json"

    def stageClaimUri(self, nRows: int, stage: StageName) -> str:
        return f"{self._tagged_prefix('results')}/{nRows}/{stage}.claim.json"

    def resourcesFor(self, stage: StageName) -> StageResources:
        return self.stageResources[stage]

    def clusterSourceFor(self, nRows: int) -> ClusterSourceRef | None:
        for item in self.clusterSources:
            if item.nRows == nRows:
                return item
        return None


def load_profiling_config(path: str | Path) -> ProfilingConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[2] / config_path
    raw = tomllib.loads(config_path.read_text())
    bucket = experiment_bucket()
    config = ProfilingConfig.model_validate(_normalize_raw_config(_bucket_uris(raw, bucket)))
    if config.clusterSources or not CLUSTER_SOURCES_PATH.is_file():
        return config
    extra = tomllib.loads(CLUSTER_SOURCES_PATH.read_text())
    refs = tuple(
        (
            ClusterSourceRef.model_validate(_cluster_source(item, bucket))
            for item in extra.get("clusterSources", [])
        )
    )
    if not refs:
        return config
    return config.model_copy(update={"clusterSources": refs})


def bind_cluster_source(
    config: ProfilingConfig, nRows: int, stages: Iterable[StageName]
) -> WorkflowParameters:
    if "importClusters" not in set(stages):
        if config.workflow.clusterSourceUri is not None:
            raise ValueError("workflow.clusterSourceUri requires the importClusters stage")
        return config.workflow
    if config.workflow.clusterSourceUri is not None:
        return config.workflow
    source = config.clusterSourceFor(nRows)
    if source is None:
        raise ValueError(f"importClusters has no cluster source for {nRows} cells")
    store_uri, artifact_id = source.resolved()
    return config.workflow.model_copy(
        update={"clusterSourceUri": store_uri, "clusterSourceArtifactId": artifact_id}
    )


def require_consume_only_override(
    config: ProfilingConfig, nRows: int, stages: Iterable[StageName]
) -> None:
    override = config.storeOverrideFor(nRows)
    blocked = [stage for stage in stages if stage not in CONSUME_STAGES]
    if override is not None and blocked:
        raise ValueError(
            f"The store override {override} for {nRows} cells is only for consume stages; refusing {', '.join(blocked)}"
        )


def _require_store_uri(uri: str, name: str) -> str:
    value = uri.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty when set")
    if not (value.startswith("s3://") or value.startswith("/") or value.startswith("file://")):
        raise ValueError(f"{name} must be an s3:// URI or a local filesystem path")
    return value


def _bucket_uris(raw: dict[str, Any], bucket: ExperimentBucket) -> dict[str, Any]:
    stated = sorted({"r2EndpointUrl", *BUCKET_KEYS.values()} & set(raw))
    if stated:
        raise ValueError(
            f"Remove {', '.join(stated)} from the configuration: it gives datasetPrefix and "
            "resultsPrefix, keys in the experiment bucket, and R2_ENDPOINT_URL and R2_BUCKET "
            "give the bucket"
        )
    missing = sorted(set(BUCKET_KEYS) - set(raw))
    if missing:
        raise ValueError(f"The configuration lacks {', '.join(missing)}")
    payload = {key: value for key, value in raw.items() if key not in BUCKET_KEYS}
    payload["r2EndpointUrl"] = bucket.endpoint
    for key, field in BUCKET_KEYS.items():
        payload[field] = bucket.uri(raw[key])
    if "clusterSources" in raw:
        payload["clusterSources"] = [
            _cluster_source(item, bucket) for item in raw["clusterSources"]
        ]
    return payload


def _cluster_source(raw: dict[str, Any], bucket: ExperimentBucket) -> dict[str, Any]:
    if "leidenRecordUri" in raw:
        raise ValueError(
            "Give a cluster source's runLeiden record as leidenRecord, its key in the experiment "
            "bucket, not as leidenRecordUri"
        )
    source = {key: value for key, value in raw.items() if key != "leidenRecord"}
    if "leidenRecord" in raw:
        source["leidenRecordUri"] = bucket.uri(raw["leidenRecord"])
    return source


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    fixed = payload.pop("fixedResources", None)
    if fixed is not None and "stageResources" not in payload:
        stages = payload.get("stages")
        selected = tuple(stages) if stages is not None else CORE_STAGE_ORDER
        payload["stageResources"] = {stage: fixed for stage in selected}
    return payload
