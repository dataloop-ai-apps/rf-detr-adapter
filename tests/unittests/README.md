# Unit Tests

## Overview

The unit tests in this repository are designed to ensure the reliability and functionality of key features, by local
model and item creation and prediction accuracy for both images and videos.

## Features

1. **Local Model and Item Creation**
    - The tests dynamically create local models and items using predefined JSON files stored in the repository.
    - The item json file and the media file expected to be with the same name.

2. **Prediction Function Testing**
    - The unit tests include comprehensive checks for the `predict` function.
    - Predictions are tested on both images and videos across all models available in the repository.

3. **Validation Using `dtlpy` Metrics**
    - The results are validated against `dtlpy` metrics and threshold values to assess accuracy.
    - Final predictions are asserted based on these thresholds.

## Recommendations

- When testing video predictions, it is recommended to use **low threshold values**.  
  This prevents test failures caused by occasional low predictions for individual frames in a video.

## Running the Tests

1. Ensure all dependencies are installed and models are correctly set up.
2. Run the test suite using your preferred test runner (e.g., `unittest` or `pytest`).
3. Run the tests from the project's root.

```bash 
python -m unittest discover -s tests/unittests -p "test_*.py"
