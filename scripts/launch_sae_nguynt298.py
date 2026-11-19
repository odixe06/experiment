"""Two authorized NILM SAE runs on newly supplied nguynt298 account."""
from launch_sae_wave2 import launch

JOBS = [('nguynt298', 'nilm', 42), ('nguynt298', 'nilm', 43)]

if __name__ == '__main__':
    launch(JOBS)
