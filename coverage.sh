# run and display coverage report properly filtered
coverage run --source=. -m unittest
coverage report -m --fail-under=80 --omit='darts/tests*,*__init__.py'