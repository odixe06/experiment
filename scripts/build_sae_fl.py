"""Thin SAE entry point for the shared single-backbone notebook builder."""
import sys

from build_tabm_fl import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--backbone", "sae"]
    main()
