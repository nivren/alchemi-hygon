#!/usr/bin/env bash
# Source this file from the project root (or use its absolute path):
#   source scripts/activate_hygon_env.sh project
#   source scripts/activate_hygon_env.sh exploration

if [[ -n ${ZSH_VERSION:-} ]]; then
    _hygon_script_path="${(%):-%N}"
else
    _hygon_script_path="${BASH_SOURCE[0]}"
fi

_hygon_sourced=0
if [[ -n ${BASH_VERSION:-} && ${BASH_SOURCE[0]} != "$0" ]]; then
    _hygon_sourced=1
elif [[ -n ${ZSH_VERSION:-} && ${ZSH_EVAL_CONTEXT:-} == *:file* ]]; then
    _hygon_sourced=1
fi

_hygon_fail() {
    echo "[hygon-env] $*" >&2
    if [[ $_hygon_sourced -eq 1 ]]; then
        return 1
    fi
    exit 1
}

_hygon_project_root="$(CDPATH= cd -- "$(dirname -- "$_hygon_script_path")/.." && pwd)"
_hygon_mode="${1:-project}"
_hygon_dtk_env="${HYGON_DTK_ENV:-/opt/dtk-26.04/env.sh}"
_hygon_uv_cache="${UV_CACHE_DIR:-/data/envs/uv-cache}"
_hygon_pypi_index="https://mirrors.bfsu.edu.cn/pypi/web/simple"

case "$_hygon_mode" in
    project)
        _hygon_python_env="$_hygon_project_root/.venv"
        ;;
    exploration)
        _hygon_python_env="/home/wangleping/codes/nvalchemi-toolkit/.venv"
        ;;
    *)
        _hygon_fail "usage: source scripts/activate_hygon_env.sh [project|exploration]"
        return $?
        ;;
esac

if [[ ! -f $_hygon_dtk_env ]]; then
    _hygon_fail "DTK environment not found: $_hygon_dtk_env"
    return $?
fi
if [[ ! -f $_hygon_python_env/bin/activate ]]; then
    _hygon_fail "Python environment not found: $_hygon_python_env"
    return $?
fi

# shellcheck source=/dev/null
source "$_hygon_dtk_env" || {
    _hygon_fail "failed to load $_hygon_dtk_env"
    return $?
}
# shellcheck source=/dev/null
source "$_hygon_python_env/bin/activate" || {
    _hygon_fail "failed to activate $_hygon_python_env"
    return $?
}

export HYGON_PROJECT_ROOT="$_hygon_project_root"
export HYGON_PYTHON_ENV="$_hygon_python_env"
export UV_CACHE_DIR="$_hygon_uv_cache"
export UV_INDEX_URL="$_hygon_pypi_index"
export UV_DEFAULT_INDEX="$_hygon_pypi_index"
export PIP_INDEX_URL="$_hygon_pypi_index"
export PYTHONPATH="$_hygon_project_root/packages/framework:$_hygon_project_root/packages/ops${PYTHONPATH:+:$PYTHONPATH}"

echo "[hygon-env] DTK: $_hygon_dtk_env"
echo "[hygon-env] Python: $_hygon_python_env"
echo "[hygon-env] PYTHONPATH: $PYTHONPATH"
echo "[hygon-env] PyPI: $UV_INDEX_URL"
echo "[hygon-env] UV cache: $UV_CACHE_DIR"

unset _hygon_script_path _hygon_sourced _hygon_project_root _hygon_mode
unset _hygon_dtk_env _hygon_uv_cache _hygon_pypi_index _hygon_python_env
