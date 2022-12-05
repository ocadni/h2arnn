
import pandas as pd


def train_net(
    net2train,
    betas,
    lr=0.001,
    max_step=500,
    batch_size=2000,
    batch_iter=10,
    std_fe_limit=1e-4,
    suffix="",
    exact=False,
    stats_step=1
):
    stats = []
    net2train.train(
        beta=betas[0],
        lr=lr,
        max_step=2000,
        batch_size=batch_size,
        std_fe_limit=std_fe_limit,
        exact=exact,
    )
    steps = 0
    for beta in betas:
        net2train.train(
            beta=beta,
            lr=lr,
            max_step=max_step,
            batch_size=batch_size,
            std_fe_limit=std_fe_limit,
            exact=exact,
            set_optim=False
        )

        if steps % stats_step == 0:
            stats.append(net2train.compute_stats(
                beta, batch_size=batch_size, batch_iter=batch_iter, print_=True))
        steps += 1

    stats_pd = pd.DataFrame(stats)
    return stats_pd.add_suffix(suffix)
