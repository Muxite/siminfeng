# Docker images removed 2026-11-12 (disk 9 GB -> 24 GB free)

All results are recorded in their experiment dirs and build records; images are rebuildable from those records.
Kept: campaign jetson-c1, jetson-up1108 (bases), om11, om12 (current best); factory bare-cf34d311 (frozen
bare reference a9a0c71c), bd173c08-pregate-6e9d96cf (061e58e7, run-4 scored best), bd173c08-restored-0b514fba
(1274abb8, run-4 final); l4t-cuda base; ubuntu:22.04.

| tag | image id |
|---|---|
| factory bd173c08-names-133b42ae | 0b3eb1cee0ac |
| factory bd173c08-fill-c0427fe6 | 05633f1d3a15 |
| factory bd173c08-router-5cded3e7 | 69f6a4169f38 |
| factory bd173c08-mmap-93a61111 | b857d53d0e6e |
| factory bd173c08-bare-cf34d3113 | 7fd50837114f |
| factory f0a3c6-bare-restored / -mmap / -bare | bd0b28ccb075 / b1a71b44d7c8 / 3a45f08b49df |
| factory fe255fce restored-bare / mmap-advice / bare | 7682b1003d5d / cd4dbf6c4a4a / 37de536a2403 |
| campaign jetson-c1-j3 / -rebuild / -p01 / -p01q / -p03 / -tr03 / -ks | 29d0041dc097 / cf8e40de68c9 / 5ded7cddc76c / ad20ebabe24f / 09075b53c619 / bec0080503a3 / 8ec53cfa6fbf |
| campaign jetson-up1108-mc / -ra / -om7 / -om8 / -om8b / -om9 / -om10 / -om10b | 69b8ae28bf49 / 289ed0622cd1 / 4d0d6e6a04ff / 6a23534cec7a / 8b987232fa1c / e93415ad9a6d / cc626c7f8592 / 916714d41389 |

Plus docker build cache (9.6 GB reclaimed).
