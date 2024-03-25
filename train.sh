game="kuhn_poker(players=3)" # specify the game
solver="cmd" # use MD-type or cfr-type algorithm
comb="gmd" # use KL/EU (closed-form solution) or GMD (numerical method)
alpha_sc="mtctl" # use meta-controller to adjust alpha or fix the alpha
alpha_optim="drs" # the meta-controller used to adjust alpha (other choices: dgld, rs, ...)
alpha_optim_obj="negap" # the evaluation metric of interest (other choices: sw, ccegap)
phi=3 # the choice of Bregman divergence/convex function
seed=1 # random seed

python train.py --game ${game} --solver ${solver} --comb ${comb} --alpha_sc ${alpha_sc} \
                --alpha_optim ${alpha_optim} --alpha_optim_obj ${alpha_optim_obj} \
                --phi ${phi} --seed ${seed}