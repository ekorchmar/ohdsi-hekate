set dotenv-load
default_opts := "-a $ATHENA_DOWNLOAD_DIR -b $BUILD_RXE_DOWNLOAD_DIR"
graph_args := "-g GGR:0034637"

test args="-v":
  PYTHONPATH=src/hekate pytest {{args}}

run args="":
  pip install -e .
  hekate {{args}} {{default_opts}} {{graph_args}}

scalene args="":
  scalene src/hekate/main.py {{args}} {{default_opts}} {{graph_args}}

profile:
  python -m cProfile -o program.prof src/hekate/main.py \
    {{default_opts}} \
    {{graph_args}}
  snakeviz program.prof
