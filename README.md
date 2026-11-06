# Student Risk ML

Predicts student dropout risk using the UCI dataset. Includes trained models, results, plots, and a dashboard.

## Run

Double-click `run_windows.bat`.

On a new Windows PC, run `setup_windows.bat` first.

## Files

- `app.py`: dashboard
- `train.py`: training code
- `data/data.csv`: dataset
- `outputs/`: models, results, and plots
- `test_pipeline.py`: tests

Models are already trained. Test accuracy: 76.4% at enrollment and 91.3% after first semester.

To retrain:

```powershell
.\.venv\Scripts\python.exe train.py --timing both
```

Dataset: Realinho et al. (2021), [UCI Predict Students' Dropout and Academic Success](https://doi.org/10.24432/C5MC89), CC BY 4.0. Uses 3,630 students with final outcomes, an 80/20 train/test split, and five-fold cross-validation. Excludes 794 still-enrolled students.

For coursework only
