#!/usr/bin/env bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc || true
export PATH="$HOME/.local/bin:$PATH"

if [ ! -f "pyproject.toml" ] && [ -d "maxdiffusion" ]; then
  cd maxdiffusion
fi

if [ ! -d ".venv" ]; then
  uv venv --python 3.12 .venv --seed
fi
source ./.venv/bin/activate

setup_args=("$@")
if [ ${#setup_args[@]} -eq 0 ]; then
  setup_args=(MODE=stable DEVICE=tpu)
fi
bash setup.sh "${setup_args[@]}"

# tpu create v6 --name v6-8-02-lihan --repo lihzha/maxdiffusion --branch tenny-dev --setup-cmd "git checkout tenny-dev && bash bash_scripts/setup.sh"
