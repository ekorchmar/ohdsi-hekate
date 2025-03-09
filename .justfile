test args="-v":
  PYTHONPATH=src/hekate pytest {{args}}

run args="-a $ATHENA_DOWNLOAD_DIR -b $BUILD_RXE_DOWNLOAD_DIR":
  pip install -e .
  hekate {{args}}

profile:
  python -m cProfile -o program.prof src/hekate/main.py
