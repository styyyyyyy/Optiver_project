# Historical linear-regression comparison

This directory records the original Phase 2C linear-stacking experiment.  The
two OOF predictions are used as features in a Ridge regression
(`alpha=0.01`). Predictions are generated with five-fold shuffled row-level
KFold (`random_state=42`).

| Model | RMSPE |
|---|---:|
| LightGBM | 0.219744 |
| MLP | 0.222723 |
| Linear stacking (Ridge) | 0.236216 |

The linear stack is included as a transparent negative result: ordinary Ridge
optimizes squared error rather than RMSPE and performs worse here than either
component model.

This is a **historical exploratory result**, not a strict nested grouped-CV
estimate. It should not be compared directly with the repository headline
score of `0.217123`. The focused reproduction script is
[`scripts/evaluate_linear_regression.py`](../../scripts/evaluate_linear_regression.py),
and the complete original Phase 2C analysis remains in
[`legacy/phase2c_ensemble.py`](../../legacy/phase2c_ensemble.py).
