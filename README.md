# experimental

> *A personal sandbox for probabilistic programming prototypes, archived experiments, and writing drafts.*

[![Julia](https://img.shields.io/badge/Julia-1.11-9558B2?logo=julia&logoColor=white)](https://julialang.org)
[![Vue 3](https://img.shields.io/badge/Vue-3-42b883?logo=vue.js&logoColor=white)](https://vuejs.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white)](https://vitejs.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

This repo is the experimental playground behind [shravangoswami.com](https://shravangoswami.com).  
It hosts iterative prototypes, design explorations, and draft research notes that aren't production-ready yet.  
Things here may be broken, half-baked, or just weird — and that is the point.

---

## DoodlePPL — Visual Bayesian Modelling

> Browser-based graphical editor for [JuliaBUGS](https://github.com/TuringLang/JuliaBUGS.jl), inspired by the original DoodleBUGS.

| Variant | Live Demo |
|---------|-----------|
| **Figma-like MultiCanvas** | [🔗 Open](https://shravanngoswamii.github.io/experimental/DoodlePPL/Figma-like-DoodleBUGS-MultiCanvas) |
| **GSoC DoodleBUGS** | [🔗 Open](https://shravanngoswamii.github.io/experimental/DoodlePPL/GSoC-DoodleBUGS) |
| **PrimeVue MultiCanvas** | [🔗 Open](https://shravanngoswamii.github.io/experimental/DoodlePPL/PrimeVue-GSoC-DoodleBUGS-MultiCanvas) |

Built with **Vue 3 + Vite + TypeScript** on the front end and a **Julia HTTP server** (JuliaBUGS runtime) on the back end.  
> ⚠️ WebKit / Safari / all iOS browsers are not supported — use Chrome, Edge, or Firefox.

---

## GSoC-Backup — Archived Snapshots

Frozen snapshots of the codebase taken at key milestones during **Google Summer of Code 2025**.  
Useful for diffing how the architecture evolved and recovering anything that got refactored away.

```
GSoC-Backup/
├── Code/          ← baseline implementation
├── DoodleBUGS/    ← mid-GSoC checkpoint
└── TEST/          ← throwaway test harness
```

---

## Other Experiments

| Folder | Description |
|--------|-------------|
| `sample/` | Astro static-site scaffold used as a template baseline |
| `JOSS-reviews/` | Draft materials for Journal of Open Source Software reviews |
| `Research-Work/` | Paper drafts: Deepfake detection, Hibiscus flower analysis, Log analysis tool |
| `mlg-format/` | Custom MLG citation/bibliography format experiments |
| `navbar/` | Julia + shell scripts to inject a shared navbar across pages |
| `OpenSource/` | Notes on open-source setup, distribution, and Gumroad extras |

---

## Tools

A single PowerShell script lives in `tools/` to help navigate the workspace:

```powershell
# List all items — nodes, vite projects, julia envs, html pages, astro pages, markdown, manifests
.\tools\list-workspace-items.ps1
```

---

## Running a DoodlePPL Variant Locally

```sh
# 1. Install front-end dependencies
cd DoodlePPL/Figma-like-DoodleBUGS-MultiCanvas
npm install

# 2. Start the Julia back-end (separate terminal)
cd runtime
julia --project=. server.jl

# 3. Start the Vite dev server
npm run dev
```

Open `http://localhost:5173` and draw your first Bayesian graph.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI framework | Vue 3 (Composition API) |
| Build tool | Vite 5 |
| Language | TypeScript |
| State management | Pinia |
| Backend runtime | Julia + HTTP.jl |
| Probabilistic engine | JuliaBUGS |
| Static site | Astro (sample/) |
| Formatting | Runic.jl (CI) |
