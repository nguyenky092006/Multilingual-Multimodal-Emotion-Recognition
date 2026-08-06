"""Frozen encoder specifications; weight extraction is deferred by design."""

from .guard import DownloadApprovalRequired, describe_encoder

__all__ = ["DownloadApprovalRequired", "describe_encoder"]
