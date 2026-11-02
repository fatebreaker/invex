# invex

An exact order-by-order (L3) replay of public market data, and the code to train and test trade-execution
policies in it.

invex was built to ask whether an execution policy needs to know which asset it trades. Microstructure invariance
says markets look alike once time, size and price are measured in units specific to each asset: its trade rate,
trade size, spread and tick. invex tests that on decisions rather than on market statistics:
- it trains reinforcement-learning policies in an exact replay of some stocks, with features and costs in a chosen
  unit system;
- it evaluates them zero-shot on stocks, years and venues they never saw;
- the evaluation runs on frozen, hashed episode tables whose access is logged, so tests can be fixed before their
  data are read.

The pieces are usable on their own: the parsers and replay for any L3 study, the execution environment for any
learner, and the evaluation tools for any paired comparison of policies.

## What is in the repository

**Replay engine** (`src/invex/`):
- one parser per source, all writing the same event schema:
  - Nasdaq TotalView-ITCH 5.0;
  - IEX DEEP+;
  - Coinbase L3, through Tardis.dev;
- a virtual-order tracker that places an order inside the real queues;
- three fill rules (exact, pessimistic, optimistic);
- day-level book checks.

**Execution task.** One order is bought or sold within a horizon, with up to ten decisions. At each the policy can
hold, join the touch, improve, cancel or cross. Whatever is left at the horizon is crossed.

**Unit systems** (`src/invex/env/units.py`, `features.py`): raw units, z-scores, Kyle–Obizhaeva units,
microstructure (MS) units, and variants with tick context or extra inputs (below).

**Learners** (`src/invex/agents/`):
- fitted Q-iteration on advantages, with LightGBM trees or a neural network;
- PPO.

`experiments/` adds learned alternatives to hand-designed units:
- invariance penalties (IRM, V-REx, GroupDRO; `inv_fqi.py`);
- a learned context encoder (`ctx_fqi.py`);
- domain randomisation in time, price and size (`time_dr.py`).

**Evaluation** (`src/invex/eval/`):
- the stock universe and its splits;
- frozen episode tables and a sealed loader that logs every read;
- paired evaluation of any set of policies on identical market paths;
- stock-cluster and two-way (stocks × days) bootstraps, equivalence tests (TOST), Holm and Benjamini–Hochberg.

**Controls that need no market data:**
- a zero-intelligence order-book market with exact scaling symmetries (`src/invex/sim/zi.py`,
  `experiments/zi_controls.py`);
- a family of torque-limited pendulums, a domain outside finance (`experiments/pendulum/pendulum.py`).

## Requirements

**Software:** Linux, Python ≥ 3.10, `pigz` and `curl`.

**Compute.** The replay and the tree learner run on CPUs, with numba and LightGBM; more cores help. A GPU is optional,
for the neural learner and PPO.

**Disk.**
- The 19 Nasdaq ITCH sample days are about 149 GB gzipped, plus room to decompress the day being built.
- An IEX DEEP+ day is about 11 GB raw and 1 GB processed.
- Coinbase is small.

## Install

```bash
python -m venv --system-site-packages ~/venvs/invex    # --system-site-packages reuses an installed torch
~/venvs/invex/bin/pip install -e '.[dev]'               # numpy<2, numba, torch, pyarrow, pandas, scipy, lightgbm, ...
source scripts/env.sh                                    # source it in every shell before running anything
python -m pytest                                         # about 100 unit and property tests, a minute or two
```

`scripts/env.sh` sets defaults you can override before sourcing it:

| variable | default | what it is |
|---|---|---|
| `INVEX_VENV` | `~/venvs/invex` | the virtual environment put on `PATH` |
| `INVEX_RAW` | `/tmp/invex_raw` | raw downloads (re-downloadable; may live on scratch disk) |
| `INVEX_DATA` | `./data` | processed event tables, episodes and transitions |
| `INVEX_DATA_BUDGET_GB` | `250` | builds refuse to grow `INVEX_DATA` beyond this |
| `NUMBA_NUM_THREADS` | `48` | threads for the replay; lower it on a smaller machine |

Two kinds of test are deselected by default:
- `python -m pytest -m slow` needs a real ITCH or Coinbase day on disk;
- `-m network` downloads from the internet.

Two tests skip because this repository has no frozen split or paper sources.

## Quick start without market data

```bash
source scripts/env.sh
python experiments/zi_controls.py --control pc1     # zero-intelligence market, exact similarity across assets
python experiments/zi_controls.py --control pc2     # the relative tick varies
python experiments/pendulum/pendulum.py check        # pendulum family: similarity check by value iteration
python experiments/pendulum/pendulum.py pilot --seeds 0 1 2
```

`pendulum.py run` fits every representation and takes a few hours on 16 threads; `pendulum.py report` summarises
it. The `carl-*` subcommands run a contextual pendulum in which similarity is only approximate.

## Market data

None of the data is in the repository. All of it is free.

**Nasdaq ITCH 5.0 sample days** (emi.nasdaq.com: 19 days, 2019–2026):
```bash
python -m invex.itch.download --days all --conns 4        # or --days 2019-01-30 ...; to $INVEX_RAW/itch
python -m invex.itch.build --day 2019-01-30                # event tables; --parts 2 for the large 2025-26 days
python -m invex.book.validate --venue itch --day 2019-01-30   # book checks, written to book_qa.json
python -m invex.eval.universe stats --day 2019-01-30       # per-stock day statistics
scripts/build_all_itch.sh                                  # all of the above for every day
```

**IEX DEEP+** (IEX HIST, one day after trading; it keeps the last twelve months). Use of the data is subject to the
IEX Historical Data Terms of Use.
```bash
scripts/build_iex.sh 2026-10-14                            # download, normalise, validate, day statistics
```

**Coinbase L3** (the free first-of-month files of Tardis.dev):
```bash
scripts/download_coinbase.sh     # 16 USD pairs, 00:00-02:00 UTC, the first day of each quarter from July 2020
scripts/build_coinbase.sh        # normalise, then the book checks
```

## From data to policies

**1. The stock split.** It reads the statistics of the training days and writes `prereg/splits_v0.json` (stratified
by listing market, relative tick and activity):
```bash
python -m invex.eval.universe split --days 2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18
```

**2. Frozen episode tables.** No policy is run. Each table is hashed and recorded in `prereg/episodes_<name>.json`.
```bash
python -m invex.eval.freeze --name val_v0 --split val --days 2019-10-30 2019-12-30
python -m invex.eval.freeze --name test_v0 --split test --days 2020-01-30 2021-07-13 --sealed
```

A table frozen with `--sealed` is refused unless the caller passes `--unseal` (or `unseal=True`) with a reason. Every
such read is appended to `prereg/access_log.jsonl`.

**3. Training.** Fitted Q-iteration branches every action on the same historical path, which gives exact one-step
targets. Training therefore needs no exploration.
```bash
python experiments/transfer.py gen --name train_v0 --split train \
    --days 2019-01-30 2019-03-27 2019-07-30 2019-08-30 2019-10-18      # branched transitions
python experiments/transfer.py fit --data train_v0 --tag zero --reps R0 R1 R2 R3 R3tick --seeds 0 1 2
python experiments/transfer.py fit --data train_v0 --tag zeronn --learner nn --device cuda --reps R0 R3tick
python experiments/ppo.py --rep R3tick --seed 0 --tag zeroppo --data train_v0 --device cuda
```

Models are written to `results/transfer/models/<tag>_<rep>_s<seed>.pkl`.

**4. Evaluation.** Every policy and the baselines (crossing immediately, posting and waiting) run on identical
episodes. The output has one row per episode and policy, with the cost in spreads and in basis points.
```bash
python experiments/transfer.py eval --episodes val_v0 --tags zero zeronn zeroppo --out val_v0
python experiments/transfer.py eval --episodes val_v0 --models 'zero_R3tick_s[0-2]' --rule pessimistic --out val_v0_pess
```

**5. Paired tests.** A spec file names families of hypotheses. Each compares two policies' per-episode costs, with
seeds averaged:
```json
{"alpha": 0.05,
 "families": {"units": {"unit": "c_spread", "bootstrap": "two_way",
   "hypotheses": [
     {"id": "H1", "a": "zero:R0", "b": "zero:R3tick", "test": "greater"},
     {"id": "H2", "a": "zero:R3", "b": "zero:R3tick", "test": "tost", "delta": 0.015}]}}}
```
```bash
python experiments/confirm.py --spec my_spec.json --family units --eval val_v0 --macro-prefix Val
python experiments/confirm_robust.py --spec my_spec.json --family units --eval val_v0   # with "estimator": "episode"
```

The rest of the spec format:
- **Policies** are named `<tag>:<rep>`, or `base:<name>` for a baseline.
- **Tests** are `less`, `greater`, `noninferior` and `tost` (with a margin `delta`).
- **Inference:** Holm's adjustment within each family, and percentile-bootstrap p-values.
- **Populations:** `min_onetick` and `max_onetick` restrict a family by how often the book sat at one tick before
  the episode.

## Representations

The training cost is per share, in the price unit of the system named in the last column.

| name | features | cost per share in |
|---|---|---|
| `R0` | raw (seconds, 100-share lots, cents) | cents |
| `R0r` | raw | spreads |
| `R0desc` | raw plus price, volume rate, volatility and tick | cents |
| `R0info` | raw plus every trailing-window statistic | spreads |
| `R0pop` | raw | a per-stock scale learned from the training data |
| `R1` | per-feature z-scores | spreads |
| `R2`, `R2tick` | Kyle–Obizhaeva units, without or with tick context | the Kyle–Obizhaeva price unit |
| `R3`, `R3tick` | MS units (mean inter-trade time, trade size and spread over the last 30 minutes), without or with tick context | spreads |
| `R3s` | MS units with the per-trade volatility as the price unit | per-trade volatility |
| `R3d` | MS units with the touch depth as the size unit | spreads |
| `Rpi`, `Rdim` | ratios within the current state; the dimensionless core only | spreads |

## How the replay is checked

- **Book consistency** on every built day (`src/invex/book/validate.py`):
  - references resolve;
  - deletes remove exactly what is left;
  - the book is rarely locked or crossed;
  - FIFO holds for at least 99% of executions;
  - rows are in time order.
- **An independent reference tracker.** A plain-Python tracker inserts the virtual order into explicit FIFO queues
  (`src/invex/sim/tracker_ref.py`). It must agree fill for fill with the fast tracker on random streams from a toy
  matching engine (`tests/test_tracker.py`).
- **Virtual twins of real orders** (`experiments/fill_validation.py`). A virtual order at a real order's queue position
  should first fill when the real one is first executed, and one placed right behind it should never fill first.

## Repository map

```
src/invex/
  schema.py, venues.py     event schema; venue scales and latencies
  itch/  iex/  coinbase/   one parser per source
  book/                    replayed quotes, day validation, trailing-window statistics
  sim/                     virtual-order tracker, reference tracker, toy matching engine, zero-intelligence market
  env/                     episodes, the execution task, features, unit systems
  agents/                  fitted Q-iteration (trees, neural), PPO
  eval/                    universe and splits, frozen episode tables, policies, statistics
  provenance.py            commit, command, versions and input hashes recorded with every result
experiments/               transitions, training, evaluation, tests and analyses
scripts/                   environment, data builds, run and queue scripts, timestamps
tests/                     parsers, tracker, execution, learners, statistics, controls
```

The `scripts/run_*.sh` and `scripts/wait_*.sh` scripts ran specific pre-registered tests. They read registration
files that are not part of this repository, and are kept as a record of how those tests were run.

Results, models and logs go to `results/` and `logs/`, which are not tracked. Every results JSON records:
- the commit, and a hash of any uncommitted change;
- the command;
- the library versions;
- the hashes of its inputs.

## Limitations

- **The replay is non-intrusive.** Historical messages do not react to the virtual order. This suits small orders of
  a few hundred shares and is wrong for large ones.
- **One venue's displayed book at a time.** There is no national best quote and no routing. A cross takes displayed
  depth only.
- **Costs are gross of fees by default.** `transfer.py gen --fees nasdaq` nets out Nasdaq's taker fee and maker
  rebate.
