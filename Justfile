set positional-arguments

test *args:
    uv run pytest "{{args}}"

datasette *args:
    uv run datasette "{{args}}"

dev *args:
    DATASETTE_SECRET=abc123 \
      uv run \
        --with ../datasette-debug-bar \
        --with ../datasette-debug-gotham \
          datasette \
            --plugins-dir sample \
            --template-dir sample/templates \
            {{args}}
