"""Importable DGP factories for spawned simulation workers."""

from src.dgp import GaussianNetwork
from src.latent_samplers import CopulaSampler, MultipleNetworksSampler


def make_network(
    network_class,
    *,
    n,
    p,
    d_x,
    d_y,
    B=None,
    rng,
    edge_var=1,
    snr=None,
    b_active_network_fraction=None,
    rdpg=False,
    latent_sampler=MultipleNetworksSampler,
    network_kwargs=None,
    **_,
):
    """Adapt runner arguments for multiple-network linear-model scenarios.

    The factory must stay in an importable module so spawned workers can
    unpickle it.
    """
    options = dict(network_kwargs or {})
    if network_class is GaussianNetwork:
        options.setdefault("edge_var", edge_var)
    else:
        options.setdefault("rdpg", rdpg)
    sampler_options = {}
    if latent_sampler is MultipleNetworksSampler:
        sampler_options.update(
            B=B,
            snr=snr,
            b_active_network_fraction=b_active_network_fraction,
        )
    elif latent_sampler is not CopulaSampler:
        raise ValueError(f"Unsupported latent sampler: {latent_sampler!r}")

    network = network_class(
        n=n,
        p=p,
        d_x=d_x,
        d_y=d_y,
        latent_sampler=latent_sampler,
        rng=rng,
        **sampler_options,
        **options,
    )
    if latent_sampler is MultipleNetworksSampler:
        # The example supplies only B=0 or B=None; snr=0 also forces zero B.
        network.is_null = B == 0 or network.snr == 0
    else:
        network.is_null = network.latent_sampler.is_null
    return network
