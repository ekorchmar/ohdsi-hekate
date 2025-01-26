import cython


@cython.cfunc
def main():
    print('Hello, World')
    code: cython.int = 0
    return code


if __name__ == '__main__':
    main()
