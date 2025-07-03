from runner.runner import HekateRunner


def _main():
    runner = HekateRunner()
    runner.run()
    runner.write_results()


if __name__ == "__main__":
    import sys

    sys.argv = "hekate \
        -a $ATHENA_DOWNLOAD_DIR \
        -b $BUILD_RXE_DOWNLOAD_DIR \
        -g GGR:3551140 \
        ".split()
    _main()
