# Legacy scripts

These files are the original exploratory pipeline preserved for historical
context. They are **not** the recommended entry points.

Known limitations include hard-coded local paths, mixed data/model/reporting
logic, target-derived cluster columns in the cached feature builder and
post-hoc ensemble evaluation. The maintained implementation is in
[`src/optiver/`](../src/optiver/) with command-line runners in
[`scripts/`](../scripts/).

The images in `figures/` are historical outputs and do not represent the final
strict nested result.
