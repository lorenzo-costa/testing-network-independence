"""Small graph-mixture experiment through the existing simulation framework.

Examples:
    python inspect_clique_mixture.py
    python inspect_clique_mixture.py --nsim 10 --nodes 50 --jobs 1
    python inspect_clique_mixture.py --methods dcorr mgc --nodes 50 100 250
    python inspect_clique_mixture.py --methods dcorr mgc ac --nodes 50 100
"""

import argparse
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.dgp import BernoulliNetwork
from src.helper_functions.simulation_functions import run_simulation
from src.methods import DistanceCorrelationTest, MultivariateACTest
from src.metrics import ReturnMetric
from src.solvers.weighted_network import ASE


# Edit the experiment settings here. Command-line options can override them.
METADATA = {
    'seed': 2,
    'nsim': 100,
    'nodes': [50, 100],
    'npermutations': 200,
    'jobs': 1,
    'methods': ['dcorr'],
    'ac_neighbors': 1,
    'output': 'results/clique_inspect',
    'alpha': .05,
    'embedding': 'ASE',
    'ties': 'inclusive',
    'pvalue': '(1 + exceedances)/(npermutations + 1)',
    'rejection': 'p < alpha',
    'confidence_intervals': '95% Wilson',
    'degeneracy': 'retained; no special p-value override',
}
ALPHA = METADATA['alpha']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsim', type=int, default=METADATA['nsim'])
    parser.add_argument('--nodes', type=int, nargs='+', default=METADATA['nodes'])
    parser.add_argument('--npermutations', type=int, default=METADATA['npermutations'])
    parser.add_argument('--seed', type=int, default=METADATA['seed'])
    parser.add_argument('--jobs', type=int, default=METADATA['jobs'])
    parser.add_argument('--methods', choices=['dcorr', 'mgc', 'ac'], nargs='+', default=METADATA['methods'])
    parser.add_argument('--ac-neighbors', type=int, default=METADATA['ac_neighbors'],
                        help='Fixed neighbor count M for unconditional AC (default: 1).')
    parser.add_argument('--output', type=Path, default=Path(METADATA['output']))
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if min(args.nsim, args.npermutations, args.jobs) < 1:
        parser.error('nsim, npermutations and jobs must be positive')
    if args.seed < 0 or min(args.nodes) < 4:
        parser.error('seed must be nonnegative and nodes must be at least 4')
    if args.ac_neighbors < 1 or args.ac_neighbors >= min(args.nodes):
        parser.error('ac-neighbors must be positive and smaller than every node count')
    if len(set(args.nodes)) != len(args.nodes) or len(set(args.methods)) != len(args.methods):
        parser.error('nodes and methods must not contain duplicates')
    return args


def build_factorial_design(args, solver=ASE):
    """The solver remains replaceable; methods share each generated graph."""
    design = []
    for n, pi in product(args.nodes, [0., .5, 1.]):
        setup = (partial(BernoulliNetwork, clique_mixture=True, pi=pi, rdpg=True), solver)
        for name in args.methods:
            if name == 'ac':
                method = partial(
                    MultivariateACTest, M=args.ac_neighbors,
                    test_type='permutation', use_permutation_coeff=False,
                    _ignore_X=True,
                )
            else:
                method = partial(DistanceCorrelationTest, test_method=name)
            design.append(dict(
                setup=setup, n=n, k=2, ky=1, pi=pi,
                method=method,
                method_label=name, alpha=ALPHA, use_true_latent=False,
                permutation_type='covariate', npermutations=args.npermutations,
                _generation_keys=('setup', 'n', 'k', 'ky'),
            ))
    return design


def summarize(trials):
    rows = []
    z = norm.ppf(.975)
    for (n, pi, method), group in trials.groupby(['n', 'pi', 'method'], sort=True):
        for label, part in [('all', group), ('B=0', group[group.B == 0]),
                            ('B=1', group[group.B == 1])]:
            m, k = len(part), int(part.reject.sum())
            if m:
                rate = k/m
                center = (rate+z*z/(2*m))/(1+z*z/m)
                radius = z*np.sqrt(rate*(1-rate)/m+z*z/(4*m*m))/(1+z*z/m)
                low, high = center-radius, center+radius
            else:
                rate = low = high = np.nan
            rows.append(dict(n=n, pi=pi, method=method, regime=label,
                             trials=m, rejections=k, rate=rate,
                             ci_low=low, ci_high=high,
                             degenerate=int(part.degenerate.sum())))
    return pd.DataFrame(rows)


def plot_rates(summary, methods, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(methods), figsize=(7*len(methods), 4.5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        for pi, label in [(0., 'Null (π=0)'), (.5, 'Mixture (π=½)'), (1., 'Coupled (π=1)')]:
            part = summary[(summary.method == method) & (summary.pi == pi)
                           & (summary.regime == 'all')]
            line, = ax.plot(part.n, part.rate, 'o-', label=label)
            ax.vlines(part.n, part.ci_low, part.ci_high, color=line.get_color())
        ax.axhline(ALPHA, color='gray', linestyle=':', label=f'{ALPHA:.1%} level')
        bound = .5 + .5*ALPHA
        ax.axhline(bound, color='black', linestyle='--', label=f'{bound:.1%} mixture bound')
        ax.set(xlabel='Nodes per graph', ylabel='Rejection rate (95% Wilson MC intervals)',
               ylim=(-.03, 1.04), title=f'ASE + {method}')
        ax.legend(fontsize=9, loc='center right')
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    print(f'Seed={args.seed}; graphs per setting={args.nsim}; '
          f'permutations per graph={args.npermutations}; alpha={ALPHA}; '
          f'AC neighbors={args.ac_neighbors}')
    results = run_simulation(
        nsim=args.nsim, factorial_design=build_factorial_design(args),
        metrics=[ReturnMetric('p-value')], rng=np.random.default_rng(args.seed),
        parallel=args.jobs > 1, n_jobs=args.jobs,
    )
    trials = pd.DataFrame([
        dict(n=r['args']['n'], pi=r['args']['pi'], method=r['args']['method_label'],
             B=r['diagnostics']['regime'], degenerate=r['diagnostics']['degenerate'],
             pvalue=r['ReturnMetric'], reject=r['ReturnMetric'] < ALPHA)
        for r in results
    ])
    summary = summarize(trials)
    # Always print overall rates; conditional rates are useful for the mixture.
    visible = summary[(summary.regime == 'all') | (summary.pi == .5)]
    print(visible.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Empty regimes have undefined rates. Degenerate graphs remain in the denominator.')
    print('P-values include ties and the plus-one correction; rejection uses p < alpha.')
    args.output.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output/'trials.csv', index=False)
    summary.to_csv(args.output/'rejection_rates.csv', index=False)
    if not args.no_plot:
        plot_rates(summary, args.methods, args.output/'power.png')
    print(f'Saved results to {args.output.resolve()}')


if __name__ == '__main__':
    main()
