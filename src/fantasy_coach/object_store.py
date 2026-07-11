"""Cloud-agnostic object download/upload for model artefacts.

The model artefact lives in blob storage and is streamed to the local
filesystem on cold start (the container image ships without a ``.joblib``).
During the GCP → AWS migration (#292) that store can be either **GCS**
(``gs://``) or **S3** (``s3://``); this module dispatches on the URI scheme so
the same env-configured URI works against either cloud with no call-site
branching. Provider SDKs are imported lazily so a deployment only needs the
SDK for the scheme it actually uses.
"""

from __future__ import annotations

from pathlib import Path

_SCHEMES = ("gs", "s3")


def parse_object_uri(uri: str) -> tuple[str, str, str]:
    """Split ``gs://bucket/key`` or ``s3://bucket/key`` into ``(scheme, bucket, key)``.

    Raises ``ValueError`` for an unsupported scheme or a missing bucket/key —
    the same validation the old GCS-only call sites did inline.
    """
    for scheme in _SCHEMES:
        prefix = f"{scheme}://"
        if uri.startswith(prefix):
            bucket, _, key = uri.removeprefix(prefix).partition("/")
            if not bucket or not key:
                raise ValueError(f"{uri!r} must be {scheme}://<bucket>/<object>")
            return scheme, bucket, key
    raise ValueError(f"unsupported object URI (want gs:// or s3://): {uri!r}")


def download_object(uri: str, dest: Path) -> None:
    """Download ``uri`` to ``dest``, creating parent dirs. Dispatches on scheme."""
    scheme, bucket, key = parse_object_uri(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if scheme == "gs":
        from google.cloud import storage  # noqa: PLC0415

        storage.Client().bucket(bucket).blob(key).download_to_filename(str(dest))
    else:  # s3
        import boto3  # noqa: PLC0415

        boto3.client("s3").download_file(bucket, key, str(dest))


def upload_object(src: Path, uri: str) -> None:
    """Upload ``src`` to ``uri``. Dispatches on scheme."""
    scheme, bucket, key = parse_object_uri(uri)
    if scheme == "gs":
        from google.cloud import storage  # noqa: PLC0415

        storage.Client().bucket(bucket).blob(key).upload_from_filename(str(src))
    else:  # s3
        import boto3  # noqa: PLC0415

        boto3.client("s3").upload_file(str(src), bucket, key)
