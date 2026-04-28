# experimental

> Personal sandbox for probabilistic programming prototypes, archived experiments, and writing drafts.

[![Julia](https://img.shields.io/badge/Julia-1.11-9558B2?logo=julia&logoColor=white)](https://julialang.org)
[![Vue 3](https://img.shields.io/badge/Vue-3-42b883?logo=vue.js&logoColor=white)](https://vuejs.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)

---

## DoodlePPL

Browser-based graphical editor for [JuliaBUGS](https://github.com/TuringLang/JuliaBUGS.jl), inspired by DoodleBUGS.

| Variant | Live |
|---------|------|
| Figma-like MultiCanvas | [open →](https://shravanngoswamii.github.io/experimental/DoodlePPL/Figma-like-DoodleBUGS-MultiCanvas) |
| GSoC DoodleBUGS | [open →](https://shravanngoswamii.github.io/experimental/DoodlePPL/GSoC-DoodleBUGS) |
| PrimeVue MultiCanvas | [open →](https://shravanngoswamii.github.io/experimental/DoodlePPL/PrimeVue-GSoC-DoodleBUGS-MultiCanvas) |

> Chrome / Edge / Firefox only — WebKit and all iOS browsers are not supported.

**Quick start**

```sh
cd DoodlePPL/Figma-like-DoodleBUGS-MultiCanvas
npm install && npm run dev          # front end → http://localhost:5173
# in a second terminal:
julia --project=runtime runtime/server.jl   # Julia back end
```

---

## Contents

| Path | What |
|------|------|
| `DoodlePPL/` | Three live variants of the DoodleBUGS app |
| `GSoC-Backup/` | Frozen GSoC 2025 snapshots (`Code/`, `DoodleBUGS/`, `TEST/`) |
| `Research-Work/` | Paper drafts — deepfake detection, hibiscus analysis, log tooling |
| `JOSS-reviews/` | JOSS review materials |
| `sample/` | Astro scaffold template |
| `navbar/` | Julia + shell scripts for shared navbar injection |
| `tools/` | `list-workspace-items.ps1` — inventory of all projects in this repo |

---

Built by [@shravanngoswamii](https://github.com/shravanngoswamii) · [Julia Slack](https://julialang.slack.com/archives/CCYDC34A0) · [shravangoswami.com](https://shravangoswami.com)
