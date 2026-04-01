# 🚀 Particle Filter (PF) Localization

This repository contains the implementation of a Particle Filter for mobile robot localization. It includes standalone modules for stochastic simulation, offline dataset processing, and real-time hardware execution.

📄 **Documentation:** The project report and demonstration video are included in the root directory.

---

## 🛠️ Execution Guide

### 1. Simulated Environment

The simulation validates the localization algorithm against a virtual ray-casting model.

> **⚠️ Important Setup:** The simulation controls are hardcoded for a larger environment. If you run it on the default map, the virtual robot will drive out of bounds.
>
> 1. Open `parameters.py`.
> 2. Comment out the current (small) map.
> 3. Uncomment the **complex map** parameters at the bottom of the file.

```bash
python simulated_pf.py
```

### 2. Offline Localization

This module runs the particle filter post-process against pre-recorded physical trials located in the `/offline_dataset` directory.

_Note: For simplicity and execution speed, the object-oriented class structure was stripped from the core filter in this script. The offline dataset utilizes fixed control commands for the entire trial, so the resulting localization visualization will display a continuous curved trajectory._

```bash
python offline_pf.py
```

### 3. Online Real-Time Localization

This script establishes a direct connection to the physical robot to run the localization filter in real-time, completely bypassing the graphical interface.

```bash
python online_pf.py
```

### 4. Integrated GUI & PD Controller

To test the closed-loop PD controller and utilize the interactive dashboard, run the GUI script.

_Note: For rapid prototyping, the PD control loop and hardware activation logic are built directly on top of the GUI script. You must activate the hardware connection from within the dashboard toggles once the web interface loads._

```bash
python new_gui_control.py
```
