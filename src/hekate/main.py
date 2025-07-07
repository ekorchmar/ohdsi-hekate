from runner.runner import HekateRunner


def _main():
    runner = HekateRunner()
    runner.run()


if __name__ == "__main__":
    import sys

    sys.argv = "hekate \
        -a $ATHENA_DOWNLOAD_DIR \
        -b $BUILD_RXE_DOWNLOAD_DIR \
        -g GGR:3551140 \
        ".split()
    _main()
