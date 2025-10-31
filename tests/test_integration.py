"""
Integration tests for the MLOps pipeline
"""
import pytest
import os
from unittest.mock import patch, Mock


class TestIntegration:
    """Integration tests"""

    def test_environment_variables(self):
        """Test that environment variables can be set"""
        test_vars = {
            "MODEL_NAME": "test-model",
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "API_BASE_URL": "http://api:8086",
        }

        for key, value in test_vars.items():
            os.environ[key] = value
            assert os.getenv(key) == value

    def test_docker_compose_services(self):
        """Test Docker Compose configuration existence"""
        docker_compose_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "docker-compose.yaml"
        )
        assert os.path.exists(docker_compose_path), "docker-compose.yaml should exist"

    def test_dockerfile_existence(self):
        """Test that all Dockerfiles exist"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        dockerfiles = [
            os.path.join(base_path, "api", "Dockerfile"),
            os.path.join(base_path, "app", "Dockerfile"),
            os.path.join(base_path, "mlflow", "Dockerfile"),
            os.path.join(base_path, "training", "Dockerfile"),
            os.path.join(base_path, "nginx", "Dockerfile"),
        ]

        for dockerfile in dockerfiles:
            assert os.path.exists(dockerfile), f"{dockerfile} should exist"

    def test_requirements_files_exist(self):
        """Test that all requirements.txt files exist"""
        base_path = os.path.dirname(os.path.dirname(__file__))
        requirements_files = [
            os.path.join(base_path, "api", "requirements.txt"),
            os.path.join(base_path, "app", "requirements.txt"),
            os.path.join(base_path, "mlflow", "requirements.txt"),
            os.path.join(base_path, "training", "requirements.txt"),
        ]

        for req_file in requirements_files:
            assert os.path.exists(req_file), f"{req_file} should exist"
