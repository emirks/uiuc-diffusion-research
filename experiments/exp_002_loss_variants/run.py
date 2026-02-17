from __future__ import annotations

import yaml


def main() -> None:
    with open("config.yaml", "r") as f:
        _cfg = yaml.safe_load(f)


if __name__ == "__main__":
    main()
