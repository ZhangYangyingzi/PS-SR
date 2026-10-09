# PS-SR results snapshot — 2026-10-09

This directory contains a read-only snapshot of metrics, audits, configuration,
and execution scripts from three local experiments. Checkpoints, source videos,
decoded videos, latent caches, and training datasets are not included.

## Main result files

- [Six-depth adapter validation history](ultraflash_state_aware_adapter_6depth_v1/metrics/val_history.csv)
- [Step3000 validation macro](ultraflash_state_aware_adapter_6depth_v1/metrics/val_step3000_macro.json)
- [Continuation training history](ultraflash_state_aware_adapter_6depth_v1/metrics/train_history_1000_3000.csv)
- [Step3000 integrity audit](ultraflash_state_aware_adapter_6depth_v1/audit/final_integrity_step3000.json)
- [Six-depth adapter final Test24 summary](ultraflash_state_aware_adapter_6depth_v1/metrics/final_raw_summary.json)
- [Blind final summary](ultraflash_state_aware_adapter_blind_final_v1/metrics/final_raw_summary.json)
- [Projection reference final summary](ultraflash_condition_projection_scale100_v1/metrics/final_raw_summary.json)

Training continuation reached global step3000. Full Val24 files for step1500,
step2000, step2500, and step3000 are included. Existing Test24 and blind final
results retain their recorded checkpoint selections; their presence in this
snapshot does not constitute evaluation of continuation checkpoints.

Scripts retain their original server paths and dependencies and are not a
standalone environment installer. The snapshot preserves the recorded numeric
results without selecting a checkpoint or interpreting the metrics.

[file_inventory.json](file_inventory.json) records the size and SHA256 of every
copied file. Source experiment files were not modified during export.
