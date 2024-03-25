import argparse
import os
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch

from algorithms.learners.cfr_learner import CFR
from algorithms.learners.md_learner import CMD
from run import main
from configs import (
    hyperparameters,
    game_choices,
    coop_games,
    single_agent_games,
    zero_sum_games,
    mix_coop_comp_games,
    objective_choices,
    solver_choices,
    cfr_variant_choices,
    alpha_optim_choices,
    alpha_optim_objs,
    comb_choices,
    alpha_schedule,
)

from utils.get_openspiel_game import get_openspiel_game

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="cmd", help="solver: cmd, cfr")
    parser.add_argument("--cfr_variant", type=str, default="standard", help="variant of cfr: standard, plus")
    parser.add_argument("--objective", type=str, default="standard", help="objective of MD")
    parser.add_argument("--opt_inter", type=int, default=10, help="update meta-controller every x iteration")
    parser.add_argument("--log_dir", type=str, default="./record/result", help="directory to save agent logs")
    parser.add_argument("--model_dir", type=str, default="./record/model", help="directory to save agent models")
    parser.add_argument("--num_updates", type=int, default=100000, help="number of updates for training")
    parser.add_argument("--num_agents", type=int, default=2, help="number of players in the game")
    parser.add_argument("--power_sc", action="store_false", default=True, help='when to log results')

    parser.add_argument("--sample_n", type=int, default=5, help="number of samples in GLD/RS")
    parser.add_argument("--R", type=float, default=0.05, help="max radius in GLD")
    parser.add_argument("--r", type=float, default=0.01, help="min radius in GLD")
    parser.add_argument("--xi", type=float, default=1, help="parameter in RS")
    parser.add_argument("--mu", type=float, default=0.01, help="fixed radius in RS")

    # parameters required to be tuned
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--game", type=str, default="kuhn_poker(players=3)", help="name of the game")
    parser.add_argument("--comb", type=str, default="kl", help="policy combination")

    parser.add_argument("--phi", type=int, default=3, choices=[1, 2, 3, 4, 5], help="convex function")
    parser.add_argument("--max_k", type=int, default=5, help="max number of history policies (K)")
    parser.add_argument("--lam_ite", type=int, default=50, help="number of iteration of Newton method")

    parser.add_argument("--alpha_sc", type=str, default="mtctl", help="whether use meta-controller")
    parser.add_argument("--alpha_optim", type=str, default="rs", help="name of meta-controller")
    parser.add_argument("--alpha_optim_obj", type=str, default="negap", help="evaluation measure")
    parser.add_argument("--k_for_e_k", type=int, default=1, help="convex function e^kx")
    parser.add_argument("--n_for_x_n", type=int, default=2, help="convex function x^n, n>1")
    parser.add_argument("--n_f", type=float, default=0.1, help="convex function -x^n, 0<=n<=1")
    parser.add_argument("--add_magnet", action="store_false", default=True, help='whether add magnet policy')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    assert args.game in game_choices
    if args.game in coop_games + list(single_agent_games.keys()):
        args.power_sc = False
        args.alpha_optim_obj = "sw"
    else:
        args.power_sc = True
    assert args.objective in objective_choices.keys()
    assert args.solver in solver_choices
    if args.solver == "cfr":
        assert args.cfr_variant in cfr_variant_choices
    assert args.alpha_optim in alpha_optim_choices
    assert args.comb in comb_choices
    assert args.alpha_sc in alpha_schedule

    game, args = get_openspiel_game(args)
    log_dir = args.log_dir
    model_dir = args.model_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = datetime.now().replace(microsecond=0)

    if args.solver == "cmd":
        hps = hyperparameters
        temp_schedule = hps[args.game]["temp_schedule"]
        lr_schedule = hps[args.game]["lr_schedule"]
        mag_lr_schedule = hps[args.game]["mag_lr_schedule"]
        args.opt_inter = hps[args.game]["opt_inter"]
        if args.alpha_optim_obj in ["negap", "sw"]:
            args.max_k = hps[args.game]["maxk"]
            args.mu = hps[args.game]["mu"]
        else:
            args.max_k = hps[args.game]["maxk_cce"]
            args.mu = hps[args.game]["mu_cce"]

        objective = objective_choices[args.objective]
        learner = CMD(
            args,
            game,
            temp_schedule,
            lr_schedule,
            mag_lr_schedule,
            objective,
        )
        optim_str = {
            "gld": f"gld_inte{args.opt_inter}_sn{args.sample_n}_R{args.R}_r{args.r}",
            "dglds": f"dglds_inte{args.opt_inter}_sn{args.sample_n}_R{args.R}_r{args.r}",
            "glds": f"glds_inte{args.opt_inter}_sn{args.sample_n}_R{args.R}_r{args.r}",
            "drs": f"drs_inte{args.opt_inter}_sn{args.sample_n}_xi{args.xi}_mu{args.mu}",
            "rs": f"rs_inte{args.opt_inter}_sn{args.sample_n}_xi{args.xi}_mu{args.mu}",
            "uniform": "unif",
        }
        opt_str, seed_str = "", ""
        if args.comb == "gmd":
            para_str = f"_li{args.lam_ite}"
            mg_str = "_add_mag" if args.add_magnet else ""
            if args.alpha_sc == "mtctl":
                assert args.alpha_optim_obj in alpha_optim_objs
                if args.game in coop_games + list(single_agent_games.keys()):
                    assert args.alpha_optim_obj in ["sw"]
                elif args.game in zero_sum_games + list(mix_coop_comp_games.keys()):
                    assert args.alpha_optim_obj not in ["sw"]
                opt_str, seed_str = f"_phi{args.phi}{para_str}_maxk{args.max_k}_alp_{optim_str[args.alpha_optim]}_obj_{args.alpha_optim_obj}{mg_str}", f"_seed{args.seed}"
            else:
                seed_str = ""
                opt_str = f"_phi{args.phi}{para_str}_maxk{args.max_k}_alp_{args.alpha_sc}{mg_str}"
        suffix = "/cmd_{}_{}_{}{}_np{}_ups{}{}".format(args.game,
                                                       args.objective,
                                                       args.comb,
                                                       opt_str,
                                                       args.num_agents,
                                                       args.num_updates,
                                                       seed_str)
        fn = (log_dir + suffix)
        mn = (model_dir + suffix)
        main(learner, args.num_updates, fn, mn, args)

    elif args.solver == "cfr":
        learner = CFR(game, args.cfr_variant == "plus", args)
        fn = (log_dir + f"/cfr_{args.cfr_variant}_{args.game}_np{args.num_agents}_ups{args.num_updates}")
        mn = (model_dir + f"/cfr_{args.cfr_variant}_{args.game}_np{args.num_agents}_ups{args.num_updates}")
        main(learner, args.num_updates, fn=fn, mn=mn, args=args)

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
