from runner.runner import HekateRunner


def _main():
    runner = HekateRunner()
    runner.run()
    runner.write_results()


if __name__ == "__main__":
    _main()
