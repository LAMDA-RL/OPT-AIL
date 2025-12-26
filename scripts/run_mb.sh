demos=1
seed=0

uv run python train_il.py env=cartpole_swingup agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=1 agent.real_ratio=0.95 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=cheetah_run agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.8 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=finger_spin agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.9 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=hopper_hop agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.8 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=walker_walk agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=1 agent.real_ratio=0.9 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=walker_run agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.8 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=walker_stand agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.85 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
sleep 2
uv run python train_il.py env=hopper_stand agent=mb_ail expert.demos="$demos" method=il method.lambda_gp=10 agent.real_ratio=0.8 agent.rollout_max_length=1 env.learn_steps=2e5 seed="$seed" agent.opt=false &
wait