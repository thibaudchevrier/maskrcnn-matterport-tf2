# CLAUDE.md

Guidance for working in this repository. Read it before changing anything.

## What this repo is

`maskrcnn-matterport`: [Matterport's Mask R-CNN](https://github.com/matterport/Mask_RCNN), ported to
TF2 graph mode in 2021 and packaged as a **model library**. It knows nothing about datasets,
experiment tracking or serving; the fashion project uses it from
[fashion-seg-train](https://github.com/thibaudchevrier/fashion-seg-train) (training, `[train]`)
and for inference (base install). Consumers pin its release wheel URL.

| Module | Content | Needs |
|--------|---------|-------|
| `mrcnn/inference.py` | Resizing, anchors, image meta, unmolding detections | numpy, scikit-image only |
| `mrcnn/serving.py` | `SavedModelPredictor`: run a `MaskRCNN.save` export | `[serve]`: any TensorFlow 2.x |
| `mrcnn/model.py` | Network, losses, data generator, `MaskRCNN` (train, detect, save) | `[train]`: TensorFlow 2.15 |
| `mrcnn/utils.py` | Boxes, masks, `Dataset` base class, AP metrics | `[train]` |
| `mrcnn/config.py` | `Config`: every setting, documented in its docstring | numpy |
| `mrcnn/visualize.py`, `mrcnn/parallel_model.py` | Plots (`[viz]`), multi-GPU | |

### Rules specific to this repo

- **Keep `mrcnn.inference` and `mrcnn.serving` free of training dependencies**: no TensorFlow
  import in `inference`, TensorFlow imported lazily in `serving`. CI runs their tests on
  Python 3.12 + numpy 2 without TensorFlow. Training code imports helpers from `mrcnn.inference`,
  never the other way round.
- **TensorFlow 2.15 / Keras 2 only for training**: the graph-mode training loop doesn't run on
  Keras 3. Import Keras as `keras` (the same package as `tf.keras` in TF 2.15).
- **The export format is a contract**: `MaskRCNN.save` writes `config.json` + a SavedModel that
  fashion-seg-train serves with `mrcnn.serving`. Changing either is a breaking change (`feat!:`).
- The license (MIT, Matterport copyright) stays in `LICENSE` and in every module docstring.
- **Documented exceptions to the shared standards, for the 2021 Matterport code only:**
  - no type annotations: types are written in the numpy docstrings instead (pydoclint is set to
    not compare them with signatures);
  - pylint's naming and complexity checks are disabled in `pyproject.toml` (math-style names
    such as `N`, `P2`, `C5`; long network-building functions kept close to upstream).

  New modules and new functions follow the shared standards in full: annotated signatures,
  descriptive names, small functions.

## Commands

```bash
make install   # uv sync --locked --all-extras (pylint needs TensorFlow and matplotlib installed)
make hooks     # once: install the pre-commit and commit-msg git hooks
make format    # ruff format + ruff --fix
make lint      # all pre-commit hooks on all files (exactly what CI runs)
make test      # smoke training/export/serving + inference tests + docstring examples
make check     # lint + test: run before every commit
```

## Standards

These standards are the same in the four repositories of the project (fashion-seg-contract,
maskrcnn-matterport-tf2, fashion-seg-train, fashion-serving). Keep them in sync.

### Environment

- **uv only** (never `pip install`): `uv add` / `uv add --dev` change dependencies and update
  `pyproject.toml` and `uv.lock` together; commit both.
- Packages from the other repositories are referenced by their **release wheel URL** in
  `[tool.uv.sources]`, like registry packages. Upgrade by changing the URL, then `uv lock`.
- Don't edit by hand: `uv.lock`, `CHANGELOG.md`, the `version` in `pyproject.toml` (commitizen owns
  the last two).
- Never commit secrets (`.dvc/*.local`, tokens) or large files (`check-added-large-files`
  blocks files over 1 MB: data and models go to DVC).

### Code quality: `make lint` = pre-commit hooks = CI

`.pre-commit-config.yaml` is the single definition of the checks. The git hooks (`make hooks`),
`make lint` and the CI lint job all run it, so a commit that passes locally passes in CI.

- **ruff format** (line length 100) and **ruff check**: pycodestyle, pyflakes, isort, pyupgrade,
  bugbear, comprehensions, simplify, and pydocstyle (numpy convention).
- **pydoclint**: every parameter, return value, yielded value, raised exception and class attribute
  is documented, with types matching the annotations.
- **pylint**: 10/10.
- Hygiene hooks: trailing whitespace, end of files, YAML/TOML syntax, merge conflicts, large files.
- A `# noqa: <code>` or `# pylint: disable=<name>` needs a reason on the same line. Never disable a
  check globally or raise a limit to make code pass: fix the code.

### Docstrings: numpy style, everywhere

Every module, class and function, public or private, has a
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring:

```python
def decode(rle: str, height: int, width: int = 1) -> np.ndarray:
    """Decode an RLE string into a boolean mask.

    Parameters
    ----------
    rle : str
        Space-separated ``start length`` pairs, 1-indexed, column-major.
    height : int
        Mask height in pixels.
    width : int
        Mask width in pixels. By default 1.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(height, width)``.

    Raises
    ------
    ValueError
        If the RLE has an odd number of values.

    Examples
    --------
    >>> decode("1 2", height=2).tolist()
    [[True], [True]]
    """
```

- Summary line in the imperative mood, ending with a period, then a blank line before sections.
- Types are written exactly like the annotations (`str | Path`, `dict[str, Any]`): pydoclint
  compares them. No `, optional` suffix: state the default in the description ("By default 1.").
- `Raises` lists the exceptions the function raises itself; mention exceptions propagated from
  callees in the description.
- Classes document their constructor parameters (`Parameters`) and attributes (`Attributes`) in the
  class docstring, not in `__init__`. Instance attributes are declared in the class body
  (`timeout: float`) so they can be checked.
- `Examples` are doctests: pytest runs them (`--doctest-modules`), so they must stay correct.
- Tests: a module docstring, and a one-line docstring per test saying which behaviour it checks.

### Code style

- Type-annotate every function signature, including private helpers.
- Import submodules explicitly (`from skimage import io, transform`), not several `import
  skimage.x` lines: linters can't tell which of those is unused. No re-export blocks: import from
  the module that defines the name.
- `pathlib.Path` over `os.path`, f-strings, no mutable default arguments, `logging` rather than
  `print` in library code, error messages that say what to do.
- Keep functions small enough for pylint's limits; split them rather than raising the limits.
- No duplicated code across repositories: shared code goes in a released package
  (fashion-seg-contract for the model's request and response, maskrcnn-matterport for Matterport
  code).

### Tests

- pytest, fast and offline by default. Every behaviour change or bug fix comes with a test.
- Tests that need data, models or services skip cleanly when they are missing (and run in CI when
  the credentials are set).
- `make check` (lint + tests) before every commit.

### Commits, PRs and releases

- [Conventional Commits](https://www.conventionalcommits.org/), checked by the `commit-msg` hook and
  on every PR: `type(scope): summary`, imperative, lower case, no final period. The body explains
  *why*. One logical change per commit.
- Types and their effect on the version (commitizen, `major_version_zero = true`):

  | Type | Release |
  |------|---------|
  | `feat` | minor |
  | `fix`, `perf`, `refactor` | patch |
  | `!` after the type, or a `BREAKING CHANGE:` footer | minor while < 1.0 (then major) |
  | `docs`, `style`, `test`, `ci`, `build`, `chore` | none |

- Work on a branch, open a PR, merge only when CI is green, with a **merge commit** (not squash: the
  individual conventional commits build the changelog). Never push to `main` directly: only the
  release workflow does (bump commit + tag).
- On merge, `release.yml` bumps the version, updates `CHANGELOG.md`, tags `vX.Y.Z` and publishes a
  GitHub Release (with the wheel and sdist for libraries).
