# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements system and user fingerprinting for analytics"""

import os
import platform
import subprocess
import sys
import uuid
from typing import Any

import cpuinfo
import google.auth
import psutil
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from loguru import logger


def user_info() -> dict[str, Any]:
    """
    Provides an ID of the currently logged in user
    """
    user_info_gathered: dict[str, Any] = {}
    try:
        user_info_gathered["userid_gcloud_cli"] = subprocess.run(
            "gcloud config get-value account",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        user_info_gathered["userid_gcloud_cli"] = None

    try:
        creds, _ = google.auth.default()
        creds.refresh(GoogleAuthRequest())
        user_info_gathered["userid_google_auth_sdk"] = getattr(
            creds, "service_account_email", None
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        user_info_gathered["userid_google_auth_sdk"] = None

    try:
        user_info_gathered["userid_computeMetadata"] = requests.get(
            (
                "http://metadata.google.internal/computeMetadata/"
                "v1/instance/service-accounts/default/email"
            ),
            headers={"Metadata-Flavor": "Google"},
            timeout=60,
        ).text
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        user_info_gathered["userid_computeMetadata"] = None

    try:
        user_info_gathered["userid_tokeninfo"] = requests.get(
            "https://www.googleapis.com/oauth2/v3/tokeninfo",
            params={
                "access_token": subprocess.run(
                    "gcloud auth print-access-token",
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
            },
            timeout=60,
        ).json()["email"]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        user_info_gathered["userid_tokeninfo"] = None

    userids = set(user_info_gathered.values())
    userids.discard(None)
    user_info_gathered["userid"] = (userids or {None}).pop()
    return user_info_gathered


def sys_info() -> dict[str, Any | None]:
    """
    Provides information about the machine running the NL2SQL Code
    To disable this data collection, create an environment variable
    called "NL2SQL_DISABLE_SYSINFO".
    """
    if "NL2SQL_DISABLE_SYSINFO" in os.environ:
        return {"SYSINFO_ENABLED": False}
    try:
        r_hostname = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/hostname",
            headers={"Metadata-Flavor": "Google"},
            timeout=60,
        )
        r_hostname.raise_for_status()
        metadata_hostname = r_hostname.text
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        metadata_hostname = None

    try:
        r_machinetype = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers={"Metadata-Flavor": "Google"},
            timeout=60,
        )
        r_machinetype.raise_for_status()
        metadata_machinetype = r_machinetype.text
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(str(exc))
        metadata_machinetype = None

    cpu_info = cpuinfo.get_cpu_info()
    mem_info = psutil.virtual_memory()
    return {
        "SYSINFO_ENABLED": True,
        **platform.uname()._asdict(),
        "node_uuid": uuid.getnode(),
        "boot_time_epoch": psutil.boot_time(),
        "python_version": cpu_info["python_version"],
        "python_build": sys.version,
        "cpu_bits": cpu_info["bits"],
        "cpu_count": cpu_info["count"],
        "cpu_model": cpu_info["brand_raw"],
        "cpu_vendor": cpu_info["vendor_id_raw"],
        "cpu_hz": cpu_info["hz_actual_friendly"],
        "ram_total_bytes": mem_info.total,
        "ram_available_bytes": mem_info.available,
        "is_colab": "google.colab" in sys.modules,
        "gcp_hostname": metadata_hostname,
        "gcp_machinetype": metadata_machinetype,
    }
