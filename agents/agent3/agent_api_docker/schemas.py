"""Contrat d'entree : rapport produit par l'agent d'analyse."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")  # tolere les champs ajoutes par l'analyseur


class Python(_Base):
    version: str
    package_manager: str = "pip"


class Package(_Base):
    name: str
    version: Optional[str] = None


class Dependencies(_Base):
    file: Optional[str] = None
    status: Literal["reliable", "incomplete", "missing", "conflicting"] = "reliable"
    packages: list[Package]
    missing_imports: list[str] = []
    api_packages_to_add: list[str] = ["fastapi", "uvicorn[standard]", "pydantic"]
    os_packages: list[str] = []


class ModelGeneration(_Base):
    required_before_serving: bool = False
    command: Optional[str] = None
    requires_files: list[str] = []
    recommended_stage: Literal["docker_build", "startup"] = "docker_build"


class ML(_Base):
    framework: Literal["sklearn", "torch", "tensorflow", "xgboost", "lightgbm", "onnx", "huggingface"]
    framework_version: Optional[str] = None
    model_file: str
    model_exists_in_project: bool = True
    model_generation: Optional[ModelGeneration] = None
    model_format: Optional[str] = None
    loader: str
    custom_classes_module: Optional[str] = None


class Entrypoint(_Base):
    file: str
    function: str
    loader_function: Optional[str] = None
    is_async: bool = False


class Inference(_Base):
    entrypoint: Entrypoint
    supports_proba: bool = False
    preprocessing_location: Literal["inside_model", "separate_code", "separate_artifact", "none"]


class Field_(_Base):
    name: str
    type: Literal["int", "float", "str", "bool"]
    required: bool = True
    allowed_values: Optional[list[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None


class OutField(_Base):
    name: str
    type: str
    optional: bool = False


class IO(_Base):
    input_type: Literal["tabular", "text", "image", "tensor"]
    input_schema: list[Field_]
    output_schema: list[OutField]
    class_labels: list[Any] = []
    example_input: Optional[dict[str, Any]] = None
    example_output: Optional[dict[str, Any]] = None
    schema_confidence: Literal["high", "medium", "low"] = "high"


class Resources(_Base):
    gpu_required: Literal["yes", "no", "optional"]
    recommended_base_image: Optional[str] = None


class EnvVar(_Base):
    name: str
    default: Optional[str] = None


class ExistingApp(_Base):
    has_api: bool = False
    port: Optional[int] = None
    has_dockerfile: bool = False
    env_vars: list[EnvVar] = []
    external_services: list[str] = []


class ProjectInfo(_Base):
    file_tree: list[str] = []
    ignorable_paths: list[str] = []
    keep_for_build_only: list[str] = []
    runtime_files: list[str] = []


class Unresolved(_Base):
    field: str
    question: str


class AnalysisReport(_Base):
    schema_version: str
    project_name: str
    needs_user_input: bool = False
    python: Python
    dependencies: Dependencies
    ml: ML
    inference: Inference
    io: IO
    resources: Resources
    existing_app: ExistingApp = Field(default_factory=ExistingApp)
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    unresolved: list[Unresolved] = []
