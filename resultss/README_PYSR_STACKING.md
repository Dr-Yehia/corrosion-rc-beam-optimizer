# Stacking -> PySR Equation Script (Where to Find It)

If you couldn't find the script, use this exact path:

- `resultss/pysr_stacking_moead_selector.py` (main script)
- `resultss/find_best_equation_from_stacking.py` (easy wrapper)

## Run on Kaggle

From repository root:

```bash
python resultss/find_best_equation_from_stacking.py --niterations 220 --populations 40 --maxsize 16
```

or directly:

```bash
python resultss/pysr_stacking_moead_selector.py --niterations 220 --populations 40 --maxsize 16
```

## Required files

- `resultss/models/model_stacking.pkl`
- `resultss/models/scaler_X.pkl`
- `resultss/models/cat_encoders.json`
- Dataset in `data/Database.csv`

## Expected outputs

- `resultss/models/pysr_stacking_metrics.json`
- `resultss/models/pysr_candidates_ranked.json`
- `resultss/equations/best_equation_stacking.txt`
- `resultss/equations/best_equation_stacking.latex`
- `resultss/figures/pysr_stacking_scatter.png`
- `resultss/figures/pysr_stacking_endpoints.png`
