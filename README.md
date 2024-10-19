# OPT-AIL

Official code for OPT-AIL: [Provably and Practically Efficient Adversarial Imitation Learning with General Function Approximation](https://openreview.net/forum?id=7YdafFbhxL).


## Usage

First, you need to install Python packages listed in `requirements.txt` using `pip install -r requirements.txt`.

The expert trajectories used during the experiments can be found here:
https://drive.google.com/drive/folders/1GiwgfrnFAjZ1JGaw3T-KeViGdOFbkRqr?usp=drive_link

Then, just run the scripts in the `scripts` dir. You can try as follows:

For dmc tasks:

```bash
sh scripts/run_dmc.sh
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

