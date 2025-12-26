from agent.mb_ail import MB_AIL
from agent.opt_ail import OPT_AIL


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]

    if args.agent.name == "opt_ail":
        print('--> Using OPT-AIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = OPT_AIL(obs_dim, action_dim, action_range, args.train.batch, args)
    elif args.agent.name == "mb_ail":
        print('--> Using MB-AIL agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = action_dim
        agent = MB_AIL(obs_dim, action_dim, action_range, args.train.batch, args)
    else:
        raise NotImplementedError

    return agent
