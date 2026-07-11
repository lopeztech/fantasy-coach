"""Tests for the cloud-agnostic object_store helpers (#292 Phase 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fantasy_coach import object_store

# ---------------------------------------------------------------------------
# parse_object_uri
# ---------------------------------------------------------------------------


def test_parse_gs_uri() -> None:
    assert object_store.parse_object_uri("gs://my-bucket/models/latest.joblib") == (
        "gs",
        "my-bucket",
        "models/latest.joblib",
    )


def test_parse_s3_uri() -> None:
    assert object_store.parse_object_uri("s3://my-bucket/models/latest.joblib") == (
        "s3",
        "my-bucket",
        "models/latest.joblib",
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/x",  # unsupported scheme
        "gs://bucket-only",  # no key
        "s3://",  # no bucket or key
        "just-a-path",
    ],
)
def test_parse_rejects_bad_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        object_store.parse_object_uri(uri)


# ---------------------------------------------------------------------------
# download_object — scheme dispatch (clients are mocked)
# ---------------------------------------------------------------------------


class _FakeBlob:
    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def download_to_filename(self, dest: str) -> None:
        self._sink["gs_dest"] = dest

    def upload_from_filename(self, src: str) -> None:
        self._sink["gs_src"] = src


class _FakeBucket:
    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def blob(self, key: str):
        self._sink["gs_key"] = key
        return _FakeBlob(self._sink)


class _FakeGcsClient:
    sink: dict = {}

    def bucket(self, name: str):
        _FakeGcsClient.sink["gs_bucket"] = name
        return _FakeBucket(_FakeGcsClient.sink)


class _FakeS3Client:
    sink: dict = {}

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        _FakeS3Client.sink.update(s3_bucket=bucket, s3_key=key, s3_dest=dest)

    def upload_file(self, src: str, bucket: str, key: str) -> None:
        _FakeS3Client.sink.update(s3_src=src, s3_bucket=bucket, s3_key=key)


@pytest.fixture
def fake_gcs(monkeypatch: pytest.MonkeyPatch):
    _FakeGcsClient.sink = {}
    import google.cloud.storage as storage

    monkeypatch.setattr(storage, "Client", _FakeGcsClient)
    return _FakeGcsClient


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch):
    _FakeS3Client.sink = {}
    import boto3

    monkeypatch.setattr(boto3, "client", lambda svc: _FakeS3Client())
    return _FakeS3Client


def test_download_dispatches_to_gcs(fake_gcs, tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "model.joblib"
    object_store.download_object("gs://bkt/models/latest.joblib", dest)
    assert dest.parent.is_dir()  # parents created
    assert fake_gcs.sink == {
        "gs_bucket": "bkt",
        "gs_key": "models/latest.joblib",
        "gs_dest": str(dest),
    }


def test_download_dispatches_to_s3(fake_s3, tmp_path: Path) -> None:
    dest = tmp_path / "sub" / "model.joblib"
    object_store.download_object("s3://bkt/models/latest.joblib", dest)
    assert dest.parent.is_dir()
    assert fake_s3.sink == {
        "s3_bucket": "bkt",
        "s3_key": "models/latest.joblib",
        "s3_dest": str(dest),
    }


def test_upload_dispatches_to_s3(fake_s3, tmp_path: Path) -> None:
    src = tmp_path / "model.joblib"
    src.write_bytes(b"x")
    object_store.upload_object(src, "s3://bkt/models/latest.joblib")
    assert fake_s3.sink == {
        "s3_src": str(src),
        "s3_bucket": "bkt",
        "s3_key": "models/latest.joblib",
    }


def test_download_rejects_bad_scheme_before_touching_clients(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        object_store.download_object("http://x/y", tmp_path / "m")
