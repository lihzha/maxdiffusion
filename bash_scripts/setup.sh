curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
uv venv --python 3.12 ~/maxdiffusion_venv --seed
source ~/maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu

# tpu create v6 --name v6-8-02-lihan --repo lihzha/maxdiffusion --branch tenny-dev --setup-cmd "git checkout tenny-dev && bash bash_scripts/setup.sh"