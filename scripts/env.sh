# Source before running anything:  source scripts/env.sh
# Raw downloads live on /tmp (1.5 TB free, wiped at boot); processed data on /data (shared, 94% full).
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INVEX_REPO="$_repo"
export INVEX_RAW="${INVEX_RAW:-/tmp/invex_raw}"
export INVEX_DATA="${INVEX_DATA:-$_repo/data}"
export INVEX_DATA_BUDGET_GB="${INVEX_DATA_BUDGET_GB:-250}"
# the machine is shared: leave headroom for other jobs
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-48}"
export NUMBA_THREADING_LAYER="${NUMBA_THREADING_LAYER:-omp}"
# numba invalidates a cached function only when its own file changes, not when a function it calls from another
# module does; so the cache is keyed by the hash of every source file, and any change compiles afresh
_src="$(cat $(find "$_repo/src/invex" -name '*.py' | sort) 2> /dev/null | sha1sum | cut -c1-12)"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$_repo/.numba_cache/$_src}"
unset _src
export OMP_WAIT_POLICY=PASSIVE
export PATH="${INVEX_VENV:-$HOME/venvs/invex}/bin:$PATH"
unset _repo
