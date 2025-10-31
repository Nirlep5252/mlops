import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from api import app, fetch_latest_model, fetch_latest_version, TITANIC_FEATURES

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints"""

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_predict_endpoint_success(self, mock_fetch_version, mock_fetch_model):
        """Test successful prediction"""
        # Mock the model loading
        mock_fetch_model.return_value = "titanic-classifier"
        
        # Mock the model prediction
        mock_model = Mock()
        mock_model.predict.return_value = [1]  # Survived
        mock_fetch_version.return_value = mock_model

        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                "sex": "female",
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "S",
            },
        )

        assert response.status_code == 200
        assert "survived" in response.json()
        assert response.json()["survived"] in [0, 1]

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_predict_endpoint_with_default_embarked(
        self, mock_fetch_version, mock_fetch_model
    ):
        """Test prediction with default embarked value"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_fetch_version.return_value = mock_model

        response = client.get(
            "/predict/",
            params={
                "pclass": 3,
                "sex": "male",
                "age": 30.0,
                "sibsp": 1,
                "parch": 2,
                "fare": 15.0,
            },
        )

        assert response.status_code == 200
        assert response.json()["survived"] in [0, 1]

    def test_predict_endpoint_invalid_sex(self):
        """Test prediction with invalid sex value"""
        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                "sex": "invalid",
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "S",
            },
        )

        assert response.status_code == 400
        assert "sex must be 'male' or 'female'" in response.json()["detail"]

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_predict_endpoint_missing_optional_param(
        self, mock_fetch_version, mock_fetch_model
    ):
        """Test prediction with missing optional parameter"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_fetch_version.return_value = mock_model

        response = client.get(
            "/predict/",
            params={
                "pclass": 2,
                "sex": "female",
                "age": 35.0,
                "sibsp": 1,
                "parch": 0,
                "fare": 25.0,
            },
        )

        assert response.status_code == 200

    def test_predict_endpoint_missing_required_param(self):
        """Test prediction with missing required parameter"""
        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                # Missing 'sex'
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
            },
        )

        assert response.status_code == 422  # Unprocessable Entity

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_predict_different_passenger_classes(
        self, mock_fetch_version, mock_fetch_model
    ):
        """Test predictions for different passenger classes"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_fetch_version.return_value = mock_model

        for pclass in [1, 2, 3]:
            response = client.get(
                "/predict/",
                params={
                    "pclass": pclass,
                    "sex": "male",
                    "age": 30.0,
                    "sibsp": 0,
                    "parch": 0,
                    "fare": 20.0,
                    "embarked": "S",
                },
            )
            assert response.status_code == 200


class TestModelFunctions:
    """Test model loading functions"""

    @patch("api.MlflowClient")
    def test_fetch_latest_model_success(self, mock_client):
        """Test successful model fetching"""
        mock_model = Mock()
        mock_model.name = "titanic-classifier"
        mock_client.return_value.get_registered_model.return_value = mock_model

        result = fetch_latest_model()
        assert result == "titanic-classifier"

    @patch("api.MlflowClient")
    def test_fetch_latest_model_not_found(self, mock_client):
        """Test model not found error"""
        from mlflow.exceptions import MlflowException

        mock_client.return_value.get_registered_model.side_effect = MlflowException(
            "Model not found"
        )

        with pytest.raises(RuntimeError) as exc_info:
            fetch_latest_model()
        assert "not found" in str(exc_info.value)

    @patch("api.mlflow.pyfunc.load_model")
    def test_fetch_latest_version_success(self, mock_load_model):
        """Test successful model version fetching"""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        result = fetch_latest_version("titanic-classifier")
        assert result == mock_model
        mock_load_model.assert_called_once_with(
            model_uri="models:/titanic-classifier/Production"
        )

    @patch("api.mlflow.pyfunc.load_model")
    def test_fetch_latest_version_failure(self, mock_load_model):
        """Test model version fetch failure"""
        mock_load_model.side_effect = Exception("Failed to load model")

        with pytest.raises(RuntimeError) as exc_info:
            fetch_latest_version("titanic-classifier")
        assert "Failed to load model" in str(exc_info.value)


class TestDataValidation:
    """Test input data validation"""

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_sex_normalization(self, mock_fetch_version, mock_fetch_model):
        """Test sex value normalization"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_fetch_version.return_value = mock_model

        # Test with uppercase
        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                "sex": "MALE",
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "S",
            },
        )
        assert response.status_code == 200

        # Test with mixed case
        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                "sex": "Female",
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "S",
            },
        )
        assert response.status_code == 200

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_embarked_normalization(self, mock_fetch_version, mock_fetch_model):
        """Test embarked value normalization"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_fetch_version.return_value = mock_model

        # Test with lowercase
        response = client.get(
            "/predict/",
            params={
                "pclass": 2,
                "sex": "male",
                "age": 30.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 20.0,
                "embarked": "c",
            },
        )
        assert response.status_code == 200

    @patch("api.fetch_latest_model")
    @patch("api.fetch_latest_version")
    def test_empty_embarked_uses_default(self, mock_fetch_version, mock_fetch_model):
        """Test that empty embarked value uses default"""
        mock_fetch_model.return_value = "titanic-classifier"
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_fetch_version.return_value = mock_model

        response = client.get(
            "/predict/",
            params={
                "pclass": 1,
                "sex": "female",
                "age": 25.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 50.0,
                "embarked": "",
            },
        )
        assert response.status_code == 200
