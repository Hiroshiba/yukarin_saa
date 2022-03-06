from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from yukarin_saa.utility import dataclass_utility
from yukarin_saa.utility.git_utility import get_branch_name, get_commit_id


@dataclass
class DatasetConfig:
    phoneme_list_glob: str
    start_accent_list_glob: str
    end_accent_list_glob: str
    start_accent_phrase_list_glob: str
    end_accent_phrase_list_glob: str
    f0_glob: str
    volume_glob: str
    f0_process_mode: str
    phoneme_type: str
    phoneme_mask_max_length: int
    phoneme_mask_num: int
    accent_mask_max_length: int
    accent_mask_num: int
    speaker_dict_path: Optional[Path]
    speaker_size: Optional[int]
    test_num: int
    test_trial_num: int = 1
    valid_phoneme_list_glob: Optional[str] = None
    valid_start_accent_list_glob: Optional[str] = None
    valid_end_accent_list_glob: Optional[str] = None
    valid_start_accent_phrase_list_glob: Optional[str] = None
    valid_end_accent_phrase_list_glob: Optional[str] = None
    valid_f0_glob: Optional[str] = None
    valid_volume_glob: Optional[str] = None
    valid_speaker_dict_path: Optional[Path] = None
    valid_trial_num: Optional[int] = None
    valid_num: Optional[int] = None
    seed: int = 0


@dataclass
class NetworkConfig:
    phoneme_size: int
    phoneme_embedding_size: int
    speaker_size: int
    speaker_embedding_size: int
    hidden_size: int
    block_num: int
    post_layer_num: int


@dataclass
class ModelConfig:
    f0_statistics_path: Optional[Path]


@dataclass
class TrainConfig:
    batch_size: int
    log_iteration: int
    eval_iteration: int
    snapshot_iteration: int
    stop_iteration: int
    optimizer: Dict[str, Any]
    weight_initializer: Optional[str] = None
    step_shift: Optional[Dict[str, Any]] = None
    noam_shift: Optional[Dict[str, Any]] = None
    num_processes: Optional[int] = None
    use_gpu: bool = True
    use_amp: bool = False
    use_multithread: bool = False


@dataclass
class ProjectConfig:
    name: str
    tags: Dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig
    network: NetworkConfig
    model: ModelConfig
    train: TrainConfig
    project: ProjectConfig

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        backward_compatible(d)
        return dataclass_utility.convert_from_dict(cls, d)

    def to_dict(self) -> Dict[str, Any]:
        return dataclass_utility.convert_to_dict(self)

    def add_git_info(self):
        self.project.tags["git-commit-id"] = get_commit_id()
        self.project.tags["git-branch-name"] = get_branch_name()


def backward_compatible(d: Dict[str, Any]):
    if "f0_statistics_path" not in d["model"]:
        d["model"]["f0_statistics_path"] = None
