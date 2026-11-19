"""Build the locked 14-method x 3-seed SAE notebook matrix locally."""
import subprocess
import sys


OWNERS = {
    "fedavg": ("dangthiet", "dangthiet", "hoangdang1711"),
    "fedprox": ("hoangdang1711", "hoangdangnhi", "hoangdangnhi"),
    "afpha": ("hoangdv1711", "hoangdv1711", "minhiw"),
    "fd_ids": ("minhiw", "phamquang0606", "phamquang0606"),
    "apfed": ("ttalgies", "ttalgies", "qwerasdf3004"),
    "lwfednids": ("dangthiet", "dangthiet", "hoangdang1711"),
    "tinyproto": ("hoangdang1711", "hoangdangnhi", "hoangdangnhi"),
    "pfedes": ("hoangdv1711", "hoangdv1711", "minhiw"),
    "nilm": ("minhiw", "phamquang0606", "phamquang0606"),
    "fedtgp": ("ttalgies", "ttalgies", "qwerasdf3004"),
    "perfedskd": ("dangthiet", "dangthiet", "hoangdang1711"),
    "fedproto": ("hoangdangnhi", "hoangdv1711", "minhiw"),
    "v2x_htfl": ("phamquang0606", "ttalgies", "qwerasdf3004"),
}


def main():
    for method, owners in OWNERS.items():
        for seed, owner in zip((41, 42, 43), owners):
            subprocess.run([sys.executable, "scripts/build_sae_fl.py", "--owner", owner,
                            "--method", method, "--seed", str(seed)], check=True)
    for seed, owner in zip((41, 42, 43), OWNERS["fedavg"]):
        parent = f"{owner}/sae-fl-fedavg-s{seed}"
        subprocess.run([sys.executable, "scripts/build_sae_fl.py", "--owner", owner,
                        "--method", "canperfl", "--seed", str(seed), "--parent", parent], check=True)


if __name__ == "__main__":
    main()
