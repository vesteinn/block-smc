#!/bin/bash
# Wrapper for VAL Validate that sets the library path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${DIR}:${LD_LIBRARY_PATH}"
exec "${DIR}/Validate" "$@"
