import pytest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from model_training import (
    train,
    TITANIC_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    REGISTERED_MODEL_NAME,
)


class TestTrainingPipeline:
    """Test model training pipeline"""

    @patch("model_training.pd.read_csv")
    @patch("model_training.train_test_split")
    @patch("model_training.mlflow.start_run")
    @patch("model_training.MlflowClient")
    @patch("model_training.mlflow.sklearn.log_model")
    def test_train_function_success(
        self,
        mock_log_model,
        mock_mlflow_client,
        mock_start_run,
        mock_train_test_split,
        mock_read_csv,
    ):
        """Test successful model training"""
        # Create mock data
        mock_data = pd.DataFrame(
            {
                "pclass": [1, 2, 3, 1],
                "sex": ["male", "female", "male", "female"],
                "age": [25, 30, 35, 40],
                "sibsp": [0, 1, 0, 1],
                "parch": [0, 0, 1, 2],
                "fare": [50, 25, 15, 100],
                "embarked": ["S", "C", "Q", "S"],
                "survived": [1, 1, 0, 1],
            }
        )
        mock_read_csv.return_value = mock_data

        # Mock train_test_split
        X = mock_data[TITANIC_FEATURES]
        y = mock_data["survived"]
        mock_train_test_split.return_value = (X, X, y, y)

        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Mock model info
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/test_run_id/model"
        mock_log_model.return_value = mock_model_info

        # Mock MLflow client
        mock_client = Mock()
        mock_mlflow_client.return_value = mock_client

        # Mock registered model
        mock_registered_model = Mock()
        mock_client.get_registered_model.return_value = mock_registered_model

        # Mock model version
        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_client.create_model_version.return_value = mock_model_version

        # Run training
        train()

        # Assertions
        mock_read_csv.assert_called_once()
        mock_train_test_split.assert_called_once()
        mock_start_run.assert_called_once()
        mock_log_model.assert_called_once()
        mock_client.create_model_version.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_once()

    @patch("model_training.pd.read_csv")
    def test_data_loading(self, mock_read_csv):
        """Test that data is loaded correctly"""
        mock_data = pd.DataFrame(
            {
                "Pclass": [1, 2, 3],
                "Sex": ["male", "female", "male"],
                "Age": [25, 30, 35],
                "SibSp": [0, 1, 0],
                "Parch": [0, 0, 1],
                "Fare": [50, 25, 15],
                "Embarked": ["S", "C", "Q"],
                "Survived": [1, 1, 0],
            }
        )
        mock_read_csv.return_value = mock_data

        with patch("model_training.train_test_split"):
            with patch("model_training.mlflow.start_run"):
                with patch("model_training.MlflowClient"):
                    with patch("model_training.mlflow.sklearn.log_model"):
                        try:
                            train()
                        except Exception:
                            pass  # We just want to test data loading

        mock_read_csv.assert_called_once()

    @patch("model_training.pd.read_csv")
    @patch("model_training.train_test_split")
    @patch("model_training.mlflow.start_run")
    @patch("model_training.MlflowClient")
    @patch("model_training.mlflow.sklearn.log_model")
    def test_model_registration(
        self,
        mock_log_model,
        mock_mlflow_client,
        mock_start_run,
        mock_train_test_split,
        mock_read_csv,
    ):
        """Test model registration in MLflow"""
        # Setup mocks
        mock_data = pd.DataFrame(
            {
                "pclass": [1, 2],
                "sex": ["male", "female"],
                "age": [25, 30],
                "sibsp": [0, 1],
                "parch": [0, 0],
                "fare": [50, 25],
                "embarked": ["S", "C"],
                "survived": [1, 1],
            }
        )
        mock_read_csv.return_value = mock_data

        X = mock_data[TITANIC_FEATURES]
        y = mock_data["survived"]
        mock_train_test_split.return_value = (X, X, y, y)

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/test_run_id/model"
        mock_log_model.return_value = mock_model_info

        mock_client = Mock()
        mock_mlflow_client.return_value = mock_client

        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_client.create_model_version.return_value = mock_model_version

        # Run training
        train()

        # Verify model registration
        mock_client.create_model_version.assert_called_once_with(
            name=REGISTERED_MODEL_NAME,
            source=mock_model_info.model_uri,
            run_id=mock_run.info.run_id,
        )

    @patch("model_training.pd.read_csv")
    @patch("model_training.train_test_split")
    @patch("model_training.mlflow.start_run")
    @patch("model_training.MlflowClient")
    @patch("model_training.mlflow.sklearn.log_model")
    def test_model_stage_transition(
        self,
        mock_log_model,
        mock_mlflow_client,
        mock_start_run,
        mock_train_test_split,
        mock_read_csv,
    ):
        """Test model stage transition to Production"""
        # Setup mocks
        mock_data = pd.DataFrame(
            {
                "pclass": [1, 2],
                "sex": ["male", "female"],
                "age": [25, 30],
                "sibsp": [0, 1],
                "parch": [0, 0],
                "fare": [50, 25],
                "embarked": ["S", "C"],
                "survived": [1, 1],
            }
        )
        mock_read_csv.return_value = mock_data

        X = mock_data[TITANIC_FEATURES]
        y = mock_data["survived"]
        mock_train_test_split.return_value = (X, X, y, y)

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/test_run_id/model"
        mock_log_model.return_value = mock_model_info

        mock_client = Mock()
        mock_mlflow_client.return_value = mock_client

        mock_model_version = Mock()
        mock_model_version.version = "2"
        mock_client.create_model_version.return_value = mock_model_version

        # Run training
        train()

        # Verify stage transition
        mock_client.transition_model_version_stage.assert_called_once_with(
            name=REGISTERED_MODEL_NAME,
            version="2",
            stage="Production",
            archive_existing_versions=True,
        )

    @patch("model_training.pd.read_csv")
    @patch("model_training.train_test_split")
    @patch("model_training.mlflow.start_run")
    @patch("model_training.MlflowClient")
    @patch("model_training.mlflow.sklearn.log_model")
    def test_model_not_registered_creates_new(
        self,
        mock_log_model,
        mock_mlflow_client,
        mock_start_run,
        mock_train_test_split,
        mock_read_csv,
    ):
        """Test that a new registered model is created if it doesn't exist"""
        from mlflow.exceptions import RestException

        # Setup mocks
        mock_data = pd.DataFrame(
            {
                "pclass": [1, 2],
                "sex": ["male", "female"],
                "age": [25, 30],
                "sibsp": [0, 1],
                "parch": [0, 0],
                "fare": [50, 25],
                "embarked": ["S", "C"],
                "survived": [1, 1],
            }
        )
        mock_read_csv.return_value = mock_data

        X = mock_data[TITANIC_FEATURES]
        y = mock_data["survived"]
        mock_train_test_split.return_value = (X, X, y, y)

        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/test_run_id/model"
        mock_log_model.return_value = mock_model_info

        mock_client = Mock()
        mock_mlflow_client.return_value = mock_client

        # Simulate model not found
        mock_client.get_registered_model.side_effect = RestException("Not found")

        mock_model_version = Mock()
        mock_model_version.version = "1"
        mock_client.create_model_version.return_value = mock_model_version

        # Run training
        train()

        # Verify create_registered_model was called
        mock_client.create_registered_model.assert_called_once_with(
            REGISTERED_MODEL_NAME
        )


class TestFeatureConfiguration:
    """Test feature configuration"""

    def test_titanic_features_list(self):
        """Test that TITANIC_FEATURES contains expected features"""
        expected_features = [
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
        ]
        assert TITANIC_FEATURES == expected_features

    def test_numeric_features_list(self):
        """Test numeric features configuration"""
        expected_numeric = ["age", "sibsp", "parch", "fare"]
        assert NUMERIC_FEATURES == expected_numeric

    def test_categorical_features_list(self):
        """Test categorical features configuration"""
        expected_categorical = ["pclass", "sex", "embarked"]
        assert CATEGORICAL_FEATURES == expected_categorical

    def test_all_features_covered(self):
        """Test that numeric and categorical features cover all Titanic features"""
        all_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        assert all_features == set(TITANIC_FEATURES)
