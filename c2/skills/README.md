# Campaign Claude Code skills

From /home/muk/projects/siminfeng, copy the five skill directories into
this project's Claude Code skills directory (review existing same-name skills
before replacing them):

```sh
mkdir -p /home/muk/projects/siminfeng/.claude/skills
cp -R c2/skills/engine-ab-experiment c2/skills/engine-build \
  c2/skills/over-memory-run c2/skills/codex-delegation c2/skills/kernel-profile \
  /home/muk/projects/siminfeng/.claude/skills/
```

These skills assume the campaign repo exists at its current absolute path;
procedural references resolve from the campaign root. Installation is a separate
operator action; the checked-in catalog remains c2/METHODS.md.

- **engine-ab-experiment** — Prepare, queue and assess a counterbalanced A/B.
- **engine-build** — Build a pinned commit/patch and preserve input identity.
- **over-memory-run** — Run guarded >DRAM inference with fair baselines and IO metrics.
- **codex-delegation** — Write bounded assignments, launch workers and check acceptance.
- **kernel-profile** — Use nsys inventories and kernel-local screens with proper limits.
