"""다운로드한 RAR에서 필요한 조건만 준비하고, 기록별 특징을 캐시한다.

학습·설정 조정·최종 평가 파일은 서로 다른 폴더와 명시적인 목록으로 관리한다.
이 모듈은 경고 기준을 학습하거나 최종 평가 신호를 자동으로 분석하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable

import pandas as pd

from modules.paderborn import extract_window_features, load_paderborn, reference_label_for_bearing
from modules.vibration_config import SPLIT_NAMES, VibrationConfig


@dataclass
class WindowedRecord:
    """원본 파일의 메타데이터와 해당 파일에서 추출한 시간 구간 특징."""

    metadata: dict
    features: pd.DataFrame


def write_json(path: Path, value: dict) -> None:
    """NaN이 들어간 JSON을 만들지 않고, 쓰기 도중 원래 파일이 깨지지 않게 한다."""
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison_dir(data_root: Path) -> Path:
    return data_root / "derived" / "comparison"


def expected_records(config: VibrationConfig) -> list[dict]:
    """파일 이름을 정해진 규칙으로 생성해 RAR의 임의 경로를 사용하지 않는다."""
    config.validate()
    records = []
    for split in SPLIT_NAMES:
        for bearing in config.splits[split]:
            for number in range(1, config.records_per_bearing + 1):
                name = f"{config.condition}_{bearing}_{number}.mat"
                records.append({
                    "file": name, "bearing_id": bearing, "record_number": number,
                    "condition": config.condition, "split": split,
                    "provided_label": reference_label_for_bearing(bearing),
                    "relative_path": f"raw/{split}/{name}",
                })
    return records


def _archive_members(archive: Path, bearing: str) -> dict[str, int]:
    """Windows 기본 tar(bsdtar)로 일반 파일의 경로와 원래 크기만 조회한다."""
    if shutil.which("tar") is None:
        raise ValueError("RAR을 읽을 수 있는 tar(bsdtar)가 필요합니다.")
    result = subprocess.run(["tar", "-tvf", str(archive)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"압축파일 목록을 읽지 못했습니다: {archive.name}\n{result.stderr}")
    members = {}
    for line in result.stdout.splitlines():
        match = re.search(r"(" + re.escape(bearing) + r"/\S+)$", line)
        if match is None:
            continue
        fields = line[:match.start()].split()
        if not fields or not fields[0].startswith("-"):
            continue  # 디렉터리·링크는 추출 대상에서 제외한다.
        if len(fields) < 5 or not fields[4].isdigit():
            raise ValueError("tar 파일 목록의 크기 형식을 해석할 수 없습니다.")
        name = match.group(1)
        if name in members:
            raise ValueError(f"압축파일 안에 중복 경로가 있습니다: {name}")
        members[name] = int(fields[4])
    return members


def prepare_dataset(archive_dir: Path, data_root: Path, config: VibrationConfig,
                    progress: Callable[[str], None] = print) -> dict:
    """선택한 조건의 MAT와 설명서만 추출한다. 이미 있는 다른 파일은 덮어쓰지 않는다."""
    records = expected_records(config)
    plans = []
    # 모든 압축파일과 필요한 기록을 먼저 확인한 뒤 추출을 시작한다.
    for split, bearings in config.splits.items():
        for bearing in bearings:
            archive = archive_dir / f"{bearing}.rar"
            if not archive.is_file():
                raise ValueError(f"압축파일이 없습니다: {archive}")
            members = _archive_members(archive, bearing)
            selected = [r for r in records if r["bearing_id"] == bearing]
            names = [f"{bearing}/{r['file']}" for r in selected]
            if any(name not in members for name in names):
                raise ValueError(f"{bearing}: 선택한 조건의 기록 1~{config.records_per_bearing}가 필요합니다.")
            documents = [f"{bearing}/{bearing}.pdf", f"{bearing}/measuring_log_{bearing}.pdf"]
            names += [name for name in documents if name in members]
            plans.append((archive, bearing, split, members, names, selected))

    raw_root = (data_root / "raw").resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    for archive, bearing, split, members, names, selected in plans:
        # 화이트리스트에 포함된 파일만 임시 폴더에 추출한다.
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=raw_root) as directory:
            staging = Path(directory).resolve()
            if not staging.is_relative_to(raw_root):
                raise ValueError("임시 추출 경로가 데이터 폴더를 벗어났습니다.")
            result = subprocess.run(["tar", "-xf", str(archive.resolve()), "-C", str(staging), *names],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                raise ValueError(f"{archive.name} 추출 실패: {result.stderr}")
            for name in names:
                source = staging / name
                if not source.is_file() or source.is_symlink() or source.stat().st_size != members[name]:
                    raise ValueError(f"추출 파일의 종류나 크기를 확인해 주세요: {name}")
                folder = raw_root / (split if name.endswith(".mat") else f"documents/{bearing}")
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / source.name
                if not target.resolve().is_relative_to(raw_root):
                    raise ValueError("추출 대상 경로가 데이터 폴더를 벗어났습니다.")
                digest = file_sha256(source)
                if target.exists():
                    if file_sha256(target) != digest:
                        raise ValueError(f"다른 내용의 기존 파일을 덮어쓸 수 없습니다: {target}")
                else:
                    source.replace(target)
                for record in selected:
                    if record["file"] == target.name:
                        record["sha256"] = digest
        progress(f"{bearing}: {config.records_per_bearing}개 준비 ({split})")

    manifest = {"schema_version": 1, "config_fingerprint": config.fingerprint,
                "config": config.to_dict(), "records": records}
    write_json(comparison_dir(data_root) / "dataset_manifest.json", manifest)
    return manifest


def load_manifest(data_root: Path, config: VibrationConfig) -> dict:
    path = comparison_dir(data_root) / "dataset_manifest.json"
    if not path.is_file():
        raise ValueError("데이터 목록이 없습니다. 먼저 prepare를 실행해 주세요.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("config_fingerprint") != config.fingerprint:
        raise ValueError("데이터 목록과 설정이 다릅니다. prepare와 train을 다시 실행해 주세요.")
    expected = expected_records(config)
    actual = manifest.get("records", [])
    if len(expected) != len(actual):
        raise ValueError("데이터 목록의 기록 수가 설정과 다릅니다.")
    for wanted, found in zip(expected, actual):
        if any(found.get(key) != value for key, value in wanted.items()) or not found.get("sha256"):
            raise ValueError("데이터 목록의 용도·파일 정보가 설정과 다릅니다.")
    return manifest


def read_windowed_record(path: Path, config: VibrationConfig,
                         cache_dir: Path | None = None) -> WindowedRecord:
    """원본 해시와 특징 설정이 같을 때만 특징 캐시를 재사용한다."""
    digest = file_sha256(path)
    cache_path = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps([digest, config.feature_settings], sort_keys=True).encode()).hexdigest()
        cache_path = cache_dir / f"{path.stem}_{key[:16]}.json"
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return WindowedRecord(cached["metadata"], pd.DataFrame(cached["windows"]))
    record = load_paderborn(path)
    features = extract_window_features(record, config.window_sec, config.hop_sec)
    metadata = {**record.metadata, "source_sha256": digest,
                "feature_settings": config.feature_settings}
    windowed = WindowedRecord(metadata, features)
    if cache_path is not None:
        # 일정한 신호의 정의 불가능한 첨도 등을 캐시에 정상값처럼 저장하지 않는다.
        if features[list(config.features)].notna().all().all():
            serializable = features.astype(object).where(pd.notna(features), None)
            write_json(cache_path, {"metadata": metadata, "windows": serializable.to_dict("records")})
    return windowed


def read_split(data_root: Path, manifest: dict, split: str, config: VibrationConfig,
               progress: Callable[[str], None] = print) -> list[WindowedRecord]:
    """명시한 용도의 파일만 읽는다. train에서는 evaluation을 호출하지 않는다."""
    records = []
    for entry in manifest["records"]:
        if entry["split"] != split:
            continue
        path = data_root / entry["relative_path"]
        record = read_windowed_record(path, config, comparison_dir(data_root) / "features")
        if record.metadata["source_sha256"] != entry["sha256"]:
            raise ValueError(f"준비 이후 원본 파일이 변경됐습니다: {path.name}")
        records.append(record)
        progress(f"{split}: {path.name}")
    return records
