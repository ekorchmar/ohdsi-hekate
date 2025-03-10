default_opts := "-a $ATHENA_DOWNLOAD_DIR -b $BUILD_RXE_DOWNLOAD_DIR"

test args="-v":
  PYTHONPATH=src/hekate pytest {{args}}

run args="":
  pip install -e .
  hekate {{args}} {{default_opts}}

scalene args="":
  scalene src/hekate/main.py {{args}} {{default_opts}}

profile:
  python -m cProfile -o program.prof src/hekate/main.py {{default_opts}}
