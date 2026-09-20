"""Importable DGP factories for spawned simulation workers."""

from src.dgp import GaussianNetwork


def make_network(
    network_class,
    *,
    n,
    p,
    d_x,
    d_y,
    B,
    rng,
    edge_var=1,
    snr=None,
    rdpg=False,
    network_kwargs=None,
    **_,
):
    """Adapt runner arguments for multiple-network linear-model scenarios.

    Keep this factory in a module, rather than defining it in a notebook:
    multiprocessing workers must be able to import it when unpickling tasks.
    """
    options = dict(network_kwargs or {})
    if network_class is GaussianNetwork:
        options.setdefault("edge_var", edge_var)
    else:
        options.setdefault("rdpg", rdpg)
    network = network_class(
        n=n, p=p, d_x=d_x, d_y=d_y, B=B, snr=snr, rng=rng, **options
    )
    # The example supplies only B=0 or B=None; snr=0 also forces zero B.
    network.is_null = B == 0 or network.snr == 0
    return network
