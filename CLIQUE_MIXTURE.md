# One-graph mixture experiment

For a small inspection run with a printed table, trial CSV, summary CSV,
and power plot (settings and metadata are in the script's `METADATA` block):

```sh
python inspect_clique_mixture.py --nsim 10 --nodes 50 --jobs 1
python inspect_clique_mixture.py --nsim 100 --nodes 50 100 --methods dcorr mgc
python inspect_clique_mixture.py --nsim 100 --nodes 50 100 --methods dcorr mgc ac
```

This inspection script defaults to seed 2 and 200 permutations. Results go
to `results/clique_inspect`; use `--output` to change it, or `--no-plot` to
skip plotting. Both methods share each graph and its estimated embedding
when run together. The printed mixture rows include each regime's denominator
and 95% Wilson interval. Small runs are checks, not precise power estimates.

`ac` selects the existing unconditional AC permutation test, with fixed
neighbor count `M=1` by default. Set `ac_neighbors` in the script's metadata
or pass `--ac-neighbors` to change it. It uses the scalar rank coefficient,
with no conditioning variable and no coordinate-permutation coefficient.

Run the existing framework from the repository root:

```sh
python -m src.run_simulation_script --config config_clique_mixture.yaml
```

The supplied configuration runs 100 graphs per setting, 199 attribute
permutations per graph, seed 20261106, node counts 50/100/250/500/1000,
and regime probabilities 0/0.5/1. For a final study increase `nsim` to
at least 10000. This full adjacency/embedding/permutation pipeline is more
expensive than the exact count shortcut in the original brief.

`CliqueMixtureSampler` draws a fresh B once per call, independent binary
attributes Y, and community labels independent of Y with probability 1-pi,
or equal to Y with probability pi. Two one-hot community coordinates make
`BernoulliNetwork(rdpg=True)` produce the two cliques exactly, with no loops.
The framework's Y corresponds to the brief's node attribute X; the framework's
X is reserved for optional conditioning variables and remains None.

The solver is supplied through the normal `setup` pair (YAML `solver: ASE`).
The sampler does not embed the graph or select a test. Keep k=2 and ky=1.
Replace the solver as usual, or change `test_method: dcorr` to `mgc` to use
the existing MGC test. The default experiment uses ASE, not diffusion embedding.
Its distance correlation helper reports the squared conventional distance
correlation; this monotone transformation does not change permutation ordering.

P-values use the existing sampled-permutation engine, including ties and the
plus-one correction. Rejection follows its existing strict p < alpha rule.
This replaces the original brief's exact hypergeometric calibration. There
is no clique-specific p=1 override: degenerate graphs are retained and passed
through the selected solver and existing test like other graphs. The
`diagnostics` output records the regime and whether either clique has fewer
than two nodes or the attribute vector is constant. These diagnostics are
not used to select the embedding or calculate the statistic.

Regime-specific results are diagnostics; the overall rejection fraction at
pi=0.5 estimates mixture power. For any test of level alpha under pi=0,
mixture power is bounded by (1-pi)*alpha+pi, hence 0.525 at alpha=0.05,
pi=0.5. This bound applies with ASE, diffusion embedding, or MGC. Finite
Monte Carlo estimates may exceed the population bound by sampling variation.
