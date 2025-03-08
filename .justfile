test args="-v":
  PYTHONPATH=src/hekate pytest {{args}}

run args="faster":
  python {{ if args == "faster" { "-O" } else { args } }} src/hekate/main.py

profile:
  python -m cProfile -o program.prof src/hekate/main.py
