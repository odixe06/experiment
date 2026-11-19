"""Five SAE jobs for the latest verified free slots."""
from launch_sae_wave2 import launch

JOBS = [('dangthiet', 'tinyproto', 43), ('hoangdang1711', 'pfedes', 41),
        ('hoangdangnhi', 'pfedes', 42), ('qwerasdf3004', 'pfedes', 43),
        ('qwerasdf3004', 'nilm', 41)]

if __name__ == '__main__':
    launch(JOBS)
