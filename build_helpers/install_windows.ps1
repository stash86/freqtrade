# vendored Wheels compiled via https://github.com/xmatthias/ta-lib-python/tree/ta_bundled_040

python -m pip install --upgrade pip
python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

pip install -U wheel "numpy<2"
pip install --only-binary ta-lib --find-links=build_helpers\ ta-lib

pip install -r requirements-dev.txt
pip install -e .
