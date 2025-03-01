## Contributing to Hekate
All contributions are welcome! Here are some guidelines to follow when contributing to Hekate.

## Branch policy
Currently, Hekate is being built by a single developer, who commits and pushed to `main` with reckless abandon. Please
contact me via `korchmar@ohdsi.org` if you would like to join the development.

## Testing

### Unit tests
TBD

### Testing environment
TBD

## Tooling
### Code style
We use `ruff` for Python code formatting. SQL scripts in `reference/` are formatted with `sqlfluff`.

It is highly recommended to install `pre-commit` and run `pre-commit install` in the root of the repository to ensure
that your code is properly formatted before committing.

### Type annotations
All Python code should be annotated with type hints. We use `basedpyright` to lint the code and ensure that all type
hints are correct.
