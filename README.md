# Object Classification Research Prototype

This repository contains a research prototype for an object classification system. The system is designed to be deployed on both cloud and edge devices, and it uses a variety of models and technologies to achieve high performance and accuracy.

## Table of Contents

- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Services](#running-the-services)
- [Deployment](#deployment)
- [License](#license)
- [References](#references)

## Architecture

The architecture of the system is described in the `architecture.drawio` file. It consists of a set of microservices that work together to provide the object classification functionality.

## Getting started

### Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Kubernetes](https://kubernetes.io/docs/tasks/tools/) (optional, for deployment)
- [uv](https://github.com/astral-sh/uv)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd object_classification_v2
   ```

2. **Create a virtual environment and install dependencies:**

   This project uses `uv` for package management. The dependencies are defined in each service's `pyproject.toml` file. To install the dependencies for all services, you can run the following command in the root directory:

   ```bash
   uv sync --all-packages
   ```

   Alternatively, you can install the dependencies for each service individually. For example, to install the dependencies for the `preprocessing` service:

   ```bash
   uv sync --package=preprocessing
   ```

### Running the Services

The services can be run individually or all at once using the `start_all_service.sh` script.

**To run all services:**

```bash
./start_all_service.sh
```

**To run individual services:**

Each service has a `run_server.sh` script that can be used to start it. For example, to start the inference service:

```bash
cd src/inference
./run_server.sh
```

## Deployment

The `deployment` directory contains scripts and configuration files for deploying the system to both cloud and edge environments. See the README files in those directories for more information.

## License

This project is licensed under the terms of the [Apache license](LICENSE).

## References

If you use our prototype for research, please cite our paper where we describe the prototype in detail:

```bibtex
@INPROCEEDINGS{nguyen_2025_sagely,
  author={Nguyen, Hong-Tri and Yuan, Liang and Nguyen, Anh-Dung and Babar, M. Ali and Truong, Hong-Linh},
  booktitle={2025 IEEE International Conference on Web Services (ICWS)},
  title={SAGELY - Context-Aware Holistic Service Policy Enforcement Across Swarm-Edge Continuum},
  year={2025},
  pages={607-617},
  doi={10.1109/ICWS67624.2025.00083}}
```

