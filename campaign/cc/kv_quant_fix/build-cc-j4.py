#!/usr/bin/env python3
"""Run the existing Build CC with a private recipe limited to four build jobs."""
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
campaign = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(campaign / "cc/build_engine"))
import build_engine

recipes = campaign / "artifacts/codexKVB/gpu-recipes-j4"
recipes.mkdir(parents=True, exist_ok=True)
for name in ("Dockerfile", "docker-bake.hcl"):
    shutil.copy2(build_engine.RECIPES / name, recipes / name)
dockerfile = recipes / "Dockerfile"
recipe = dockerfile.read_text()
assert recipe.count("--parallel 10") == 1, "Recheck the Build CC recipe's job limit"
dockerfile.write_text(recipe.replace("--parallel 10", "--parallel 4"))
build_engine.RECIPES = recipes
raise SystemExit(build_engine.main())
