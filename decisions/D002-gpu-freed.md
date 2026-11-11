# D002 — GPU freed with explicit user approval
- User messages (mid-session): "you may free the GPU"; "nsight can be modified, approve changes"; "you may bring down the euaaa service"; "bring down sunshine too, it hasnt been used in a while".
- Holders: euaaa-llm-gpu-1 (llama-server Qwen3.5-9B, 7.6 GB), sunshine (user systemd unit, 229 MiB).
- Actions (via run_op, evidence/orchestrator.jsonl): `docker stop euaaa-llm-gpu-1` (container kept; exit 137 = SIGKILLed after the stop timeout, so llama-server did not exit on SIGTERM in 10 s);
  `systemctl --user stop app-dev.lizardbyte.app.Sunshine.service` (not disabled).
- Result: GPU 31 MiB / 16303 MiB used, no compute apps.
- NOT touched: other euaaa containers (they depend on the LLM container and may log errors), ollama*, supabase, etc.
- RESTORE: `docker start euaaa-llm-gpu-1` (restart=unless-stopped; compose project euaaa, wd /home/muk/projects/euaaa/deploy; inspect dump in artifacts/restore/); `systemctl --user start app-dev.lizardbyte.app.Sunshine.service`.
- Nsight: /proc/driver/nvidia/params shows RmProfilingAdminOnly: 1. Approved to change, but applying needs driver reload/reboot (not done automatically). Plan: try per-container CAP_SYS_ADMIN first; only if that fails, ask before reboot.
