# Object Classification Research Prototype

This repository contains a research prototype for an object classification system. The system is designed to be deployed on both cloud and edge devices, and it uses a variety of models and technologies to achieve high performance and accuracy.

## Table of Contents

- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Services](#running-the-services)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)

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
   uv pip install -r src/ensemble/pyproject.toml -r src/inference/pyproject.toml -r src/preprocessing/pyproject.toml -r src/util/pyproject.toml
   ```

   Alternatively, you can install the dependencies for each service individually. For example, to install the dependencies for the `inference` service:

   ```bash
   cd src/inference
   uv pip install -r pyproject.toml
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

## Usage

The primary way to interact with the system is through the API exposed by the gateway service. The API endpoints and their usage are documented in the respective service directories.

## Deployment

The `deployment` directory contains scripts and configuration files for deploying the system to both cloud and edge environments. See the README files in those directories for more information.

## License

This project is licensed under the terms of the [Apache license](LICENSE).

