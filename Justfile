set positional-arguments

test *args:
    uv run pytest "$@"

datasette *args:
    uv run datasette "$@"

dev *args:
    DATASETTE_SECRET=abc123 \
      uv run \
        --with ../datasette-debug-bar \
        --with ../datasette-debug-gotham \
          datasette \
            --plugins-dir sample \
            --template-dir sample/templates \
            {{args}}

dev2 *args:
    DATASETTE_SECRET=abc123 \
    OPENROUTER_MODEL_FILTER=gemini \
      uv run \
        --with ../datasette-debug-gotham \
          datasette \
            --plugins-dir sample2 \
            --template-dir sample2/templates \
            {{args}}
