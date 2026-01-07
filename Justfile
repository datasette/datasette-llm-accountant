set positional-arguments

test *args:
    uv run pytest "$@"

datasette *args:
    uv run datasette "$@"
