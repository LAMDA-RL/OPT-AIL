# OPT-AIL

Official code for OPT-AIL: [Provably and Practically Efficient Adversarial Imitation Learning with General Function Approximation](https://openreview.net/forum?id=7YdafFbhxL).

## Quickstart

### Prerequisites
- Python version compatible with this project (see `requires-python` in `pyproject.toml`)
- `uv` installed

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and install dependencies
```bash
git clone https://github.com/LAMDA-RL/OPT-AIL.git
cd OPT-AIL

uv sync
```


## Usage
The expert trajectories used during the experiments can be found here:
https://drive.google.com/file/d/1ZBZPjQITGLkiKusdGAwObLdomTTU3bqs/view?usp=drive_link

Set `buffer_folder` and `model_folder` in `conf/config.yaml` using absolute path.

Then, just run the scripts in the `scripts` dir. You can try as follows:

Run model-free opt-ail:
```bash
sh scripts/run_mf.sh
```

Run model-based opt-ail:
```bash
sh scripts/run_mb.sh
```


## Citation

If you find this repository useful for your research, please cite:

```
@inproceedings{
	xu2024provably,
	title={Provably and Practically Efficient Adversarial Imitation Learning with General Function Approximation},
	author={Tian Xu, Zhilong Zhang, Ruishuo Chen, Yihao Sun, and Yang Yu},
	booktitle={The 38th Conference on Neural Information Processing System},
	year={2024},
	url={https://openreview.net/forum?id=7YdafFbhxL}
}
```

