from algorithms.tree import Objective

game_choices = [
    # single agent
    "single_agent_kuhn_a",
    "single_agent_kuhn_b",
    "single_agent_goofspiel",

    # cooperative
    "tiny_hanabi_a",
    "tiny_hanabi_b",
    "tiny_hanabi_c",

    # zero-sum
    "kuhn_poker(players=3)",
    "leduc_poker(players=2)",
    "goofspiel(players=3)",

    # general-sum
    "bargaining(max_turns=2)",
    "trade_comm(num_items=2)",
    "battleship",

    # mix
    "mix_kuhn_a",
    "mix_kuhn_b",
    "mix_goofspiel",
]

single_agent_games = {
    "single_agent_kuhn_a": 0,
    "single_agent_kuhn_b": 1,
    "single_agent_goofspiel": 0,
}

coop_games = [
    "tiny_hanabi_a",
    "tiny_hanabi_b",
    "tiny_hanabi_c",
]

zero_sum_games = [
    "kuhn_poker(players=3)",
    "leduc_poker(players=2)",
    "goofspiel(players=3)",
]

mix_coop_comp_games = {
    "mix_kuhn_a": [[0, 1], [2]],
    "mix_kuhn_b": [[0, 2], [1]],
    "mix_goofspiel": [[0, 1], [2]],
}

objective_choices = {
    "standard": Objective.standard,
    "maxent": Objective.maxent,
    "minimaxent": Objective.minimaxent,
}

cfr_variant_choices = ["standard", "plus"]

solver_choices = ["cmd", "cfr"]

comb_choices = ["kl", "eu", "gmd"]

alpha_optim_choices = ["drs", "dglds", "rs", "glds", "gld"]

alpha_optim_objs = ["negap", "sw", "ccegap"]

alpha_schedule = ["uniform", "mtctl", "linear_decay", "inv_sqrt_root"]

hyperparameters = {
    # single agent
    "single_agent_kuhn_a": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "single_agent_kuhn_b": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "single_agent_goofspiel": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },

    # cooperative
    "tiny_hanabi_a": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "tiny_hanabi_b": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "tiny_hanabi_c": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },

    # zero-sum
    "kuhn_poker(players=3)": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 5,
        "mu": 0.01,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "leduc_poker(players=2)": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 3,
        "mu": 0.05,
        "maxk_cce": 3,
        "mu_cce": 0.05,
        "opt_inter": 10,
    },
    "goofspiel(players=3)": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 3,
        "mu": 0.01,
        "maxk_cce": 3,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },

    # general-sum
    "bargaining(max_turns=2)": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 5,
        "mu": 0.05,
        "maxk_cce": 5,
        "mu_cce": 0.05,
        "opt_inter": 10,
    },
    "trade_comm(num_items=2)": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.01,
        "maxk_cce": 1,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "battleship": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.05,
        "maxk_cce": 5,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },

    # mix
    "mix_kuhn_a": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.01,
        "maxk_cce": 1,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "mix_kuhn_b": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.01,
        "maxk_cce": 1,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
    "mix_goofspiel": {
        "temp_schedule": lambda i: 1,
        "lr_schedule": lambda i: 0.1,
        "mag_lr_schedule": lambda i: 0.05,
        "maxk": 1,
        "mu": 0.01,
        "maxk_cce": 1,
        "mu_cce": 0.01,
        "opt_inter": 10,
    },
}