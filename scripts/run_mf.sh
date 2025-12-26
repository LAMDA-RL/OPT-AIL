demos=1
seed=0

uv run python train_il.py env=cartpole_swingup agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=1 seed="$seed" &
sleep 2
uv run python train_il.py env=cheetah_run agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
sleep 2
uv run python train_il.py env=finger_spin agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
sleep 2
uv run python train_il.py env=hopper_hop agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
sleep 2
uv run python train_il.py env=walker_walk agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=1 seed="$seed" &
sleep 2
uv run python train_il.py env=walker_run agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
sleep 2
uv run python train_il.py env=walker_stand agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
sleep 2
uv run python train_il.py env=hopper_stand agent=opt_ail expert.demos="$demos" method=il method.lambda_gp=10 seed="$seed" &
wait