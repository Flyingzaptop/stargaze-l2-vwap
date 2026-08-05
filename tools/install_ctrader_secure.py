from __future__ import annotations

from importlib.metadata import version
import re
import subprocess
import sys


MINIMUM_VERSIONS = {
    "protobuf": (6, 33, 5),
    "pyOpenSSL": (26, 0, 0),
    "requests": (2, 33, 0),
    "Twisted": (26, 4, 0),
}


def numeric_version(value: str) -> tuple[int, ...]:
    numbers = tuple(int(part) for part in re.findall(r"\d+", value))
    if not numbers:
        raise ValueError(f"version has no numeric components: {value!r}")
    return numbers


def verify_secure_runtime() -> dict[str, str]:
    installed = {name: version(name) for name in MINIMUM_VERSIONS}
    for name, minimum in MINIMUM_VERSIONS.items():
        if numeric_version(installed[name]) < minimum:
            raise RuntimeError(
                f"{name} {installed[name]} is below security floor "
                f"{'.'.join(map(str, minimum))}"
            )

    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol  # noqa: F401
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOASubscribeDepthQuotesReq,
    )

    message = ProtoOASubscribeDepthQuotesReq()
    message.ctidTraderAccountId = 1
    message.symbolId.append(2)
    restored = ProtoOASubscribeDepthQuotesReq.FromString(message.SerializeToString())
    if restored.ctidTraderAccountId != 1 or list(restored.symbolId) != [2]:
        raise RuntimeError("cTrader protobuf roundtrip failed")
    return installed


def main() -> int:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "ctrader-open-api==0.9.2",
        ],
        check=True,
    )
    installed = verify_secure_runtime()
    print(
        "secure cTrader runtime verified: "
        + ", ".join(f"{name}={value}" for name, value in installed.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
