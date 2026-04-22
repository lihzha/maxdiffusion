curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
rm -rf ~/maxdiffusion_venv
uv venv --python 3.12 ~/maxdiffusion_venv --seed
source ~/maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu