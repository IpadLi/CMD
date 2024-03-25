import pyspiel


def get_openspiel_game(args):

    # ====== single agent ======
    if args.game in [
        "single_agent_kuhn_a",
        "single_agent_kuhn_b"
    ]:
        game = pyspiel.load_game("kuhn_poker", {"players": 2})
        args.num_agents = 1
    elif args.game == "single_agent_goofspiel":
        env_configs = {
            "players": 2,
            "num_cards": 3,
            "imp_info": True,
            "points_order": "descending",
        }
        game = pyspiel.load_game_as_turn_based("goofspiel", env_configs)
        args.num_agents = 1

    # ====== cooperative ======
    elif args.game in [
        "tiny_hanabi_a",
        "tiny_hanabi_b",
        "tiny_hanabi_c",
    ]:
        # default game a
        env_configs = {
            "num_players": 2,
            "num_chance": 2,
            "num_actions": 3,
            "payoff": "10;0;0;4;8;4;10;0;0;0;0;10;4;8;4;0;0;10;0;0;10;4;8;4;0;0;0;10;0;0;4;8;4;10;0;0",
        }
        if args.game == "tiny_hanabi_game_b":
            env_configs["num_chance"] = 2
            env_configs["num_actions"] = 2
            env_configs["payoff"] = "1;0;1;0;0;1;0;1;0;1;0;0;1;0;1;0"
        elif args.game == "tiny_hanabi_game_c":
            env_configs["num_chance"] = 2
            env_configs["num_actions"] = 2
            env_configs["payoff"] = "3;0;1;3;3;0;3;0;3;2;0;2;0;1;0;0"
        game = pyspiel.load_game_as_turn_based("tiny_hanabi", env_configs)
        args.num_agents = 2

    # ====== zero-sum ======
    elif args.game == "kuhn_poker(players=3)":
        game = pyspiel.load_game("kuhn_poker", {"players": 3})
        args.num_agents = 3
    elif args.game == "leduc_poker(players=2)":
        game = pyspiel.load_game("leduc_poker", {"players": 2})
        args.num_agents = 2
    elif args.game == "goofspiel(players=3)":
        env_configs = {
            "players": 3,
            "num_cards": 3,
            "imp_info": True,
            "points_order": "descending",
        }
        game = pyspiel.load_game_as_turn_based("goofspiel", env_configs)
        args.num_agents = 3

    # ====== general-sum ======
    elif args.game in [
        "bargaining(max_turns=2)",
        "trade_comm(num_items=2)",
    ]:
        game = pyspiel.load_game(args.game)
        args.num_agents = 2
    elif args.game == "battleship":
        env_configs = {
            "loss_multiplier": 0.5,  # general-sum
            "board_width": 2,
            "board_height": 2,
            "ship_sizes": "[1]",
            "ship_values": "[1.5]",
            "num_shots": 2,
        }
        game = pyspiel.load_game(args.game, env_configs)
        args.num_agents = 2

    # ====== mix ======
    elif args.game in [
        "mix_kuhn_a",
        "mix_kuhn_b",
    ]:
        game = pyspiel.load_game("kuhn_poker", {"players": 3})
        args.num_agents = 3
    elif args.game == "mix_goofspiel":
        env_configs = {
            "players": 3,
            "num_cards": 3,
            "imp_info": True,
            "points_order": "descending",
        }
        game = pyspiel.load_game_as_turn_based("goofspiel", env_configs)
        args.num_agents = 3

    else:
        raise ValueError(f'Unknown game {args.game}')

    return game, args
