import pickle
import time

from algorithms.utils import schedule


def main(
    learner,
    num_iterations,
    fn=None,
    mn=None,
    args=None,
) -> None:
    exploit, social_welfare, ccegap = [], [], []
    ite = []
    alps = []
    info = learner.log_info()
    ite.append(1)
    exploit.append(info["Exploitability"][0])
    social_welfare.append(info["Social_Welfare"][0])
    ccegap.append(info["CCE_Gap"][0])
    if args.solver == "umd":
        alps.append(info["alpha_k"][0])
    if fn is not None:
        pickle.dump(
            {"exploit": exploit,
             "socialw": social_welfare,
             "ccegap": ccegap,
             "ite": ite,
             "alphas": alps,
             "time": 0},
            open(fn + ".pik", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        "Env {}, N {}, Iter {}/{}, NEGap {:.10f}, CCEGap {:.10f}, SW {:.4f}, KL {:.10f}, T {:.7f}s, T-mt {:.7f}s, Solver {}, Seed {}".format(
            args.game,
            args.num_agents,
            0,
            num_iterations,
            info["Exploitability"][0],
            info["CCE_Gap"][0],
            info["Social_Welfare"][0],
            info["kl"][0],
            learner.ite_time_pl,
            learner.ite_time_mt,
            args.solver,
            args.seed,
        )
    )
    start_train = time.time()
    for i, should_save in schedule(num_iterations, args.power_sc):
        learner.update()
        if should_save:
            info = learner.log_info()
            ite.append(i + 2)
            exploit.append(info["Exploitability"][0])
            social_welfare.append(info["Social_Welfare"][0])
            ccegap.append(info["CCE_Gap"][0])
            if args.solver == "umd":
                alps.append(info["alpha_k"][0])
            if fn is not None:
                pickle.dump(
                    {"exploit": exploit,
                     "socialw": social_welfare,
                     "ccegap": ccegap,
                     "ite": ite,
                     "alphas": alps,
                     "time": time.time() - start_train},
                    open(fn + ".pik", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(
                "Env {}, N {}, Iter {}/{}, NEGap {:.10f}, CCEGap {:.10f}, SW {:.4f}, KL {:.10f}, T {:.7f}s, T-mt {:.7f}s, Solver {}, Seed {}".format(
                    args.game,
                    args.num_agents,
                    i + 1,
                    num_iterations,
                    info["Exploitability"][0],
                    info["CCE_Gap"][0],
                    info["Social_Welfare"][0],
                    info["kl"][0],
                    learner.ite_time_pl,
                    learner.ite_time_mt,
                    args.solver,
                    args.seed,
                )
            )
    if mn is not None:
        pickle.dump(
            {"policy": learner.test_policy().to_dict(),
             "magnet": learner.magnet.to_dict() if args.solver == "umd" else None},
            open(mn + "_policies.pik", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )