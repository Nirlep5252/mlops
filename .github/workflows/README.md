# GitHub Actions CI/CD Pipeline

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### CI/CD Pipeline (`ci-cd.yml`)

This workflow runs on every push and pull request to the main/master/develop branches.

#### Jobs:

1. **Test** - Runs unit and integration tests
   - Tests API endpoints
   - Tests model training pipeline
   - Tests integration between components
   - Generates code coverage reports
   - Runs on Python 3.10 and 3.11

2. **Lint** - Code quality checks
   - Runs flake8 for code linting
   - Checks code formatting with black
   - Validates import sorting with isort

3. **Docker Build** - Builds Docker images
   - Tests that all Docker images build successfully
   - Uses Docker layer caching for faster builds
   - Validates API, App, Training, and MLflow images

4. **Security Scan** - Security vulnerability scanning
   - Runs Trivy security scanner
   - Checks for critical and high severity vulnerabilities
   - Uploads results to GitHub Security tab

## Local Testing

To run tests locally before pushing:

```bash
# Install dependencies
pip install -r requirements-dev.txt
pip install -r api/requirements.txt
pip install -r training/requirements.txt

# Run all tests
pytest

# Run tests with coverage
pytest --cov=api --cov=training

# Run specific test file
pytest api/test_api.py -v

# Run linting
flake8 api/api.py training/model_training.py
black --check .
isort --check-only .
```

## Required Secrets

No secrets are required for the basic CI/CD pipeline. If you add deployment stages, you may need:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- Cloud provider credentials (AWS, Azure, GCP) if deploying to cloud

## Badge

Add this badge to your README.md:

```markdown
![CI/CD Pipeline](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ci-cd.yml/badge.svg)
```
