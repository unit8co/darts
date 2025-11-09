echo "cleaning docs/build" &&
rm -rf docs/build &&
echo "cleaning docs/source/examples" &&
rm -rf docs/source/examples &&
rm -rf docs/source/generated_api &&
rm -rf docs/source/quickstart &&
rm -rf docs/source/userguide &&
rm -rf docs/source/release_notes &&
rm -rf docs/source/README.rst &&
sphinx-build -b html docs/source docs/build
