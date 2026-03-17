# Bikeshare Project

A machine learning project for predicting bike-sharing demand using a Random Forest Regressor. The project features a production-ready ML pipeline with custom feature engineering transformers, data validation, model persistence, and CI/CD via GitHub Actions.

## Features

- **End-to-End ML Pipeline** -- Scikit-learn Pipeline with imputation, mapping, outlier handling, one-hot encoding, scaling, and regression
- **Custom Transformers** -- WeekdayImputer, WeathersitImputer, Mapper, OutlierHandler, and WeekdayOneHotEncoder
- **Data Validation** -- Input validation using Pydantic schemas before prediction
- **Model Persistence** -- Trained models saved/loaded via joblib with versioning
- **Configuration Driven** -- YAML-based configuration for features, hyperparameters, and mappings
- **Automated Testing** -- Unit tests for features and prediction accuracy with pytest
- **CI/CD Pipeline** -- GitHub Actions workflow for linting (pylint), formatting (black), training, and testing

## Tech Stack

- **Language:** Python 3.10
- **ML Framework:** Scikit-learn 1.3.0
- **Data Processing:** Pandas, NumPy 1.23.2
- **Configuration:** StrictYAML, Ruamel YAML
- **Validation:** Pydantic
- **Model Serialization:** Joblib
- **Code Quality:** Black (formatting), Pylint (linting)
- **Testing:** Pytest
- **CI/CD:** GitHub Actions

## Setup / Installation

### Prerequisites

- Python 3.10

### Installation

```bash
git clone https://github.com/bnarasimha21/bikeshare-project.git
cd bikeshare-project
pip install -r requirements/requirements.txt
```

## Usage

### Train the Model

```bash
python bikeshare_model/train_pipeline.py
```

This will:
1. Load the bike rental dataset
2. Split into train/test sets (80/20)
3. Fit the pipeline (imputation, mapping, outlier handling, encoding, scaling, Random Forest)
4. Save the trained model to `bikeshare_model/trained_models/`

### Make Predictions

```python
from bikeshare_model.predict import make_prediction

data = {
    "dteday": ["2012-11-05"],
    "season": ["winter"],
    "hr": ["6am"],
    "holiday": ["No"],
    "weekday": ["Mon"],
    "workingday": ["Yes"],
    "weathersit": ["Mist"],
    "temp": [6.1],
    "atemp": [3.0014],
    "hum": [49.0],
    "windspeed": [19.0012],
}

result = make_prediction(input_data=data)
print(result["predictions"])
```

### Run Tests

```bash
pytest
```

### Lint and Format

```bash
black bikeshare_model/*.py
pylint --disable=R,C --extension-pkg-whitelist='pydantic' bikeshare_model/
```

## Project Structure

```
bikeshare-project/
├── bikeshare_model/
│   ├── config.yml              # Feature, hyperparameter, and mapping config
│   ├── config/core.py          # Config loading logic
│   ├── datasets/               # Training dataset (bike-rental-dataset.csv)
│   ├── pipeline.py             # Scikit-learn pipeline definition
│   ├── predict.py              # Prediction interface
│   ├── train_pipeline.py       # Training script
│   ├── processing/
│   │   ├── features.py         # Custom transformers
│   │   ├── data_manager.py     # Data loading and model persistence
│   │   └── validation.py       # Input validation
│   ├── tests/                  # Unit and integration tests
│   └── trained_models/         # Saved model artifacts
├── requirements/requirements.txt
└── .github/workflows/          # CI/CD pipeline
```

## License

MIT
