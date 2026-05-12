#!/usr/bin/env bash
# Wrapper de chay human.py voi dung LD_LIBRARY_PATH va PXR_PLUGINPATH_NAME
# cho omni.anim.graph.schema hoat dong dung trong standalone script.

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../../.." &> /dev/null && pwd )"

EXTSCACHE=/home/hongquan/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache

SCHEMA_EXT="$EXTSCACHE/omni.anim.graph.schema-107.3.3+107.3.3.lx64.r.cp311.u353"
USD_LIBS="$EXTSCACHE/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin"
PYTHON_LIB="/home/hongquan/miniconda3/envs/env_isaaclab/lib"

export LD_LIBRARY_PATH="$SCHEMA_EXT/lib:$USD_LIBS:$PYTHON_LIB:${LD_LIBRARY_PATH:-}"
export PXR_PLUGINPATH_NAME="$SCHEMA_EXT/plugins/AnimGraphSchema/resources:${PXR_PLUGINPATH_NAME:-}"

echo "[ENV] LD_LIBRARY_PATH prepended: $SCHEMA_EXT/lib"
echo "[ENV] PXR_PLUGINPATH_NAME: $SCHEMA_EXT/plugins/AnimGraphSchema/resources"

exec "$REPO_ROOT/isaaclab.sh" -p "$REPO_ROOT/source/isaaclab_assets/isaaclab_assets/uav/open-loop/human.py" "$@"
