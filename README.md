# Early Autism Detection AI

An AI-powered application for early autism detection using graph representation learning and deep neural networks, based on fMRI data analysis.

## Features

- Graph-based analysis of fMRI data
- Deep neural network for autism detection
- User-friendly web interface for medical professionals
- Integration with ABIDE dataset
- Using AI API's (ChatGPT and Deepseek like models to train and create a chat interface for users)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/backend/app.py
```

## Project Structure

- `src/frontend/`: React-based web interface
- `src/backend/`: Flask API and ML models
- `data/`: Directory for ABIDE dataset and preprocessed data
- `models/`: Trained model weights and configurations

## Data Source

This project uses the Autism Brain Imaging Data Exchange (ABIDE) dataset, which contains resting-state fMRI data from individuals with ASD and typical controls.

## Model Architecture

The system uses a combination of:
- Graph representation learning for fMRI connectivity analysis
- Deep neural networks for classification
- Advanced preprocessing techniques for fMRI data

## Note

This tool is intended to assist medical professionals in early autism detection. It should not be used as the sole basis for diagnosis.
