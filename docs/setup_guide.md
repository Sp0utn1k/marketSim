# Setup Guide

This guide will help you set up the Market Simulation project.

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/marketSim.git
    cd src
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Edit the `config/config.yaml` file to set up your simulation parameters and data paths.

## Running the Simulation

To run the simulation, execute the following command:
```bash
python scripts/run_simulation.py
