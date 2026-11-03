"""진동 실험 설정을 읽고, 데이터 누수가 생길 수 있는 설정을 미리 검사한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from modules.paderborn import CONDITIONS, reference_label_for_bearing


FEATURE_LABELS = {
    "rms_centered": "RMS (구간 평균 제거)",
    "kurtosis_pearson": "첨도 (Pearson)",
    "peak_to_peak": "최대·최소 차이",
    "crest_factor": "파고율",
}
SPLIT_NAMES = ("reference", "tuning", "evaluation")


@dataclass(frozen=True)
class VibrationConfig:
    """학습·분석·평가가 함께 사용하는 설정. 경로는 설정과 별도로 관리한다."""

    condition: str
    records_per_bearing: int
    window_sec: float
    hop_sec: float
    features: tuple[str, ...]
    minimum_consecutive_windows: int
    calibration_quantile: float
    n_estimators: int
    random_state: int
    splits: dict[str, tuple[str, ...]]

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError("지원하지 않는 Paderborn 운전 조건입니다.")
        if not np.isfinite([self.window_sec, self.hop_sec]).all():
            raise ValueError("구간 길이와 이동 간격은 유한한 수여야 합니다.")
        if not 2 / 64000 <= self.window_sec or not 0 < self.hop_sec <= self.window_sec:
            raise ValueError("2샘플 이상의 구간과 0 < 이동 간격 <= 구간 길이가 필요합니다.")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("분석 특징은 비어 있거나 중복될 수 없습니다.")
        if set(self.features) - set(FEATURE_LABELS):
            raise ValueError("지원하지 않는 진동 특징이 있습니다.")
        for value in (self.records_per_bearing, self.minimum_consecutive_windows, self.n_estimators):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("기록 수·연속 구간 수·트리 수는 양의 정수여야 합니다.")
        if not 0 < self.calibration_quantile < 1:
            raise ValueError("경계 설정 백분위수는 0과 1 사이여야 합니다.")
        if set(self.splits) != set(SPLIT_NAMES):
            raise ValueError("reference, tuning, evaluation을 모두 지정해야 합니다.")
        seen = set()
        for split, bearings in self.splits.items():
            if not bearings:
                raise ValueError(f"{split} 베어링 목록이 비어 있습니다.")
            for bearing in bearings:
                if not re.fullmatch(r"K(?:\d{3}|[AIB]\d{2})", bearing):
                    raise ValueError(f"잘못된 베어링 ID: {bearing}")
                if bearing in seen:
                    raise ValueError(f"베어링이 여러 용도에 중복됐습니다: {bearing}")
                label = reference_label_for_bearing(bearing)
                if label == "unknown":
                    raise ValueError(f"제공 라벨을 확인할 수 없습니다: {bearing}")
                if split == "reference" and label != "healthy":
                    raise ValueError("정상 기준에는 정상 베어링만 사용할 수 있습니다.")
                seen.add(bearing)
        if not any(reference_label_for_bearing(b) == "healthy" for b in self.splits["tuning"]):
            raise ValueError("경계 설정에 쓸 tuning 정상 베어링이 필요합니다.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """설정 변경 후 예전 모델이 조용히 재사용되는 것을 방지한다."""
        encoded = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def feature_settings(self) -> dict:
        return {"window_sec": self.window_sec, "hop_sec": self.hop_sec,
                "features": list(self.features), "extractor_version": 1}


def load_vibration_config(path: str | Path) -> VibrationConfig:
    """YAML의 항목을 명시적으로 옮겨 설정과 코드의 대응을 쉽게 읽도록 한다."""
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    try:
        forest = values["isolation_forest"]
        config = VibrationConfig(
            condition=values["condition"],
            records_per_bearing=values["records_per_bearing"],
            window_sec=float(values["window_sec"]),
            hop_sec=float(values["hop_sec"]),
            features=tuple(values["features"]),
            minimum_consecutive_windows=values["minimum_consecutive_windows"],
            calibration_quantile=float(values["calibration_quantile"]),
            n_estimators=forest["n_estimators"],
            random_state=forest["random_state"],
            splits={key: tuple(items) for key, items in values["splits"].items()},
        )
        config.validate()
        return config
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("진동 설정 파일의 항목과 형식을 확인해 주세요.") from error
