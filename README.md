# carla-bench2drive-parallel

Parallel, Gymnasium-style environments for **Bench2Drive** on **CARLA (0.9.x)**.

Built for scalable rollouts (e.g., VeRL/GRPO), a stable client-server split, and reproducible evaluation. Learn usage from the notebooks.

## Highlights

* Parallel environments with a vectorized API
* Bench2Drive scenarios for short, skill-focused routes
* Client-server split for multi-instance CARLA runs
* Config-driven (towns, routes, weather, ports, sensors)
* Tutorial notebooks that walk you from zero to rollouts

## Repository Structure

```
carla-bench2drive-parallel/
├─ client/            # Gymnasium env wrappers, rollout control, evaluators
├─ server/            # CARLA launchers, route loaders, orchestration
├─ configs/           # YAML/JSON for envs, ports, towns, weather, seeds, sensors
├─ notebooks/         # Hands-on tutorials (start here)
├─ tests/             # Smoke tests for server/env setup
├─ debug_env_init.py  # Local sanity-check helper
└─ README.md
```

## Requirements

* OS: Linux recommended (Ubuntu 20.04+)
* Python: 3.10+
* GPU: NVIDIA recommended
* CARLA: 0.9.x installed and runnable
* Bench2Drive assets/tools for evaluation

## Getting Started

Open the notebooks in the `notebooks/` folder and follow them in order. They cover: starting CARLA, connecting single/parallel envs, running short rollouts, and evaluating with Bench2Drive.

## Acknowledgements

Thanks to the Bench2Drive and CARLA communities for the simulator, scenarios, and evaluation tooling.
