# Network Intrusion Detection

This project uses [Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection) from Kaggle and aims to classify a connection as Normal or Intrusion/Anomalous.
The Dataset consists of a wide variety of intrusions simulated in a military network environment or a typical US Airforce LAN.

Deployment is done on Azure Portal, [click here](https://networkintrusiondetection.azurewebsites.net/) to visit the deployed api.

This project mainly utilizes following tools and libraries :

- Scikit Learn and Xgboost (for models and preprocessing)
- MLFLOW and Dagshub (for Experiment Tracking)
- DVC (for pipeline versioning)
- FastAPI (for server)
- Docker (for containerization)
- ACR (for Docker image registration)
- Azure Web App for Containers (for running the container)

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Structure

The project follows a modular structure for better organization and maintainability. Here's an overview of the directory structure:

- `.github/workflows`: GitHub Actions workflows for CI/CD.
- `src/`: Source code directory.
  - `NetworkIntrusionDetection/`
    - `components/`: Modules for different stages of the pipeline.
    - `utils/`: Utility functions.
    - `config/`: Configuration for each of the components.
    - `pipeline/`: Scripts for pipeline stages.
    - `entity/`: Data entity classes.
    - `constants/`: Constants used throughout the project.
- `config/`: Base Configuration for each stage og the project.
- `research/`: Directory for trials and experiments in jupyter notebook.
- `app.py`: FastAPI server.
- `Dockerfile`: Docker configuration for containerization.
- `requirements.txt`: Project dependencies.
- `setup.py`: Setup script for installing the project.
- `main.py`: Main script for execution of the complete pipeline.

## Setup

To set up the project environment, follow these steps:

1. Clone this repository.
2. Install Python 3.8 and ensure pip is installed.
3. Install project dependencies using `pip install -r requirements.txt`.
4. Ensure Docker is installed if you intend to use containerization.

## Usage

### To directly run the complete Data ingestion, Data cleaning, Model preparation and training and Model evaluation pipeline

run the command

```bash
dvc init
dvc repro
```

### To explicitly run each pipeline follow following commands-

#### Data Ingestion

To download and save the dataset, run:

```bash
python src/NetworkIntrusionDetection/pipeline/stage_01_data_ingestion.py
```

#### Preprocessing the Data

To preprocess and save the cleaned data, run:

```bash
python src/NetworkIntrusionDetection/pipeline/stage_02_eda_and_feature_engineering.py
```

#### Model Preparation and Training

To train the model, execute:

```bash
python src/NetworkIntrusionDetection/pipeline/stage_02_prepare_model.py
python src/NetworkIntrusionDetection/pipeline/stage_03_training_model.py
```

#### Model Evaluation

To evaluate the trained model, run:

```bash
python src/NetworkIntrusionDetection/pipeline/stage_04_evaluation.py
```

### To start the FastAPI server for making prediction :

Change the port to 8080 in app.py file and then,

```bash
python app.py
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

Please ensure that your contributions adhere to the project's coding standards.
