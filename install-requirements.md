# Setup Instructions for Svarog Engine Data Extraction

To run the data extraction scripts, follow these steps to set up your Python environment.

## 1. Prerequisites

Ensure you have `python3` and `python3-venv` installed on your system.

If you are on Ubuntu/Debian, you might need:
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

## 2. Environment Setup

Run the following commands in the root directory of `svarog-engine`:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

## 3. Install Dependencies

Once the virtual environment is activated, install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** You may need to run `pip install -r requirements.txt` again if you just added `python-dotenv` to the requirements file.

## 4. Project Dependencies (requirements.txt)

The following packages are required:
- `requests`: For API calls to OpenTopography and Overpass.
- `numpy`: For numerical processing of heightmaps.
- `pillow`: For image processing (saving/manipulating DEM).
- `geopandas` (optional/future): For advanced geospatial operations.
- `shapely` (optional/future): For polygon operations.

## 5. Verification and Running Tests

You can verify the installation by running:

```bash
python -c "import requests; import numpy; import PIL; print('Setup successful!')"
```

To run the tests, ensure you have your virtual environment activated and set the `PYTHONPATH` so that the `src` directory is importable:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 tests/test_extraction.py
```

