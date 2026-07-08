brew install libomp

# libomp is installed keg-only and is not symlinked into the default Homebrew
# prefix, so neither LightGBM, PyTorch, ... will find it automatically. All
# libraries bundle their own libomp.dylib; when two OpenMP runtimes are loaded
# in the same process on macOS ARM64 the dynamic linker deadlocks.
#
# DYLD_LIBRARY_PATH: makes macOS resolve "libomp.dylib" to the single shared
# Homebrew copy before the per-wheel bundled copies, so only one runtime is
# loaded at all.
LIBOMP_LIB="$(brew --prefix libomp)/lib"
echo "DYLD_LIBRARY_PATH=${LIBOMP_LIB}:${DYLD_LIBRARY_PATH}" >> "$GITHUB_ENV"
