#!/usr/bin/env bash

SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"
TOX_DIR=${SCRIPT_PATH}/../.tox
PYENV_VERSION_DIR=${HOME}/.pyenv/versions

# check that required tools are installed
NEEDED_TOOLS=(pyenv tox)
for TOOL in "${NEEDED_TOOLS[@]}"; do
    hash ${TOOL} >/dev/null 2>&1 || { echo >&2 "${TOOL} is required but it's not installed. Aborting."; exit 1; }
done


# setup multiple python versions
PYTHON_VERSIONS=(3.5.7 3.6.9 3.7.4)
for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
    echo "Setting pyenv python version ${PYTHON_VERSION}";
    pyenv install -s ${PYTHON_VERSION}
    PATH=${PYENV_VERSION_DIR}/${PYTHON_VERSION}/bin:${PATH}
done


# run tox: execute tests on multiple python versions
#tox -p auto -q
tox
