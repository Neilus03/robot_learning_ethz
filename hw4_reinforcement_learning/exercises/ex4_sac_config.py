SAC_PARAMETERS = {
    "seed": 42,
    "hidden_sizes": [256, 128, 128],
    "total_iterations": 500,  # total number of training iterations
    "learning_start_steps": 1000,  # env steps before training
    "train_freq": 500,  # env steps between SAC updates (one iteration)
    "gradient_steps": 100,  # gradient updates per SAC iteration
    "batch_size": 512,
    "eval_freq": 2048,  # env steps between evaluations / tensorboard logs
    "replay_size": 1_000_000,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "init_alpha": 0.2,
    "target_entropy": None,
    "save_interval": 10,  # save checkpoint every N eval steps (iter_<it>.pt)
}
