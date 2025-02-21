python -m pip install --upgrade build

# build darts
python -m build

# build darts, swapping the setup_darts.py files temporarily
cp setup.py setup_darts.py
cp setup_u8darts.py setup.py
python -m build
mv setup_darts.py setup.py
