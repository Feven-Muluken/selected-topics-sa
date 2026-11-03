"""Download the required TSPLIB instances without using an optimization solver."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master"
INSTANCES = {
    "att48": f"{BASE_URL}/att48.tsp",
    "ch130": f"{BASE_URL}/ch130.tsp",
    "a280": f"{BASE_URL}/a280.tsp",
}


def download_instance(name: str, destination: Path) -> None:
    url = INSTANCES[name]
    with urlopen(url, timeout=30) as response:
        content = response.read()
    text = content.decode("utf-8")
    if "NODE_COORD_SECTION" not in text or "EOF" not in text:
        raise ValueError(f"Downloaded file does not look like TSPLIB data: {url}")
    destination.write_text(text, encoding="utf-8")
    print(f"downloaded {name} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in INSTANCES:
        destination = args.output / f"{name}.tsp"
        download_instance(name, destination)


if __name__ == "__main__":
    main()
