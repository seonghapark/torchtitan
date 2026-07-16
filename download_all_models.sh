#!/usr/bin/env bash

set -uo pipefail

revisions=(
    main
)

ROOT_DIR="./gemma-7b"
mkdir "${ROOT_DIR}"
for rev in "${revisions[@]}"; do
    echo "===================================================="
    echo "Downloading ${rev}"
    echo "===================================================="

    hf download \
        google/gemma-7b \
        --revision "${rev}" \
        --local-dir "${ROOT_DIR}/${rev}"

done

