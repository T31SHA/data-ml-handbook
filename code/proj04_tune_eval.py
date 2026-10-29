# HIDE
import warnings; warnings.filterwarnings("ignore"); import sys; sys.path.insert(0, "../figures/src")
from figstyle import *
exec(open("proj03_model.py").read().split("cv = KFold")[0])
# ENDHIDE
import optuna, joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
optuna.logging.set_verbosity(optuna.logging.WARNING)
def objective(trial):
    m = make(lgb.LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        subsample=0.8, subsample_freq=1, verbose=-1, random_state=0))
    return -cross_val_score(m, X_tr, y_tr, cv=3, scoring="neg_root_mean_squared_error").mean()
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=25)
print("best CV RMSE:", round(study.best_value, 4)); print("best params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.best_params.items()})
final = make(lgb.LGBMRegressor(**study.best_params, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)).fit(X_tr, y_tr)
pred = final.predict(X_te)                                                 # test set touched ONCE
print(f"TEST  RMSE={root_mean_squared_error(y_te, pred):.4f}  MAE={mean_absolute_error(y_te, pred):.4f}  R²={r2_score(y_te, pred):.4f}")
print(f"=> typical error ≈ ${mean_absolute_error(y_te, pred)*100_000:,.0f} per block group")
pi = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=0, scoring="neg_root_mean_squared_error")
print(pd.Series(pi.importances_mean, X_te.columns).sort_values(ascending=False).round(3).head(6).to_string())
joblib.dump(final, "data/house_model.joblib")
# HIDE
res = y_te - pred
f, axs = plt.subplots(1, 3, figsize=(12, 3.6))
axs[0].scatter(y_te, pred, s=3, alpha=.3, color=T); axs[0].plot([0, 5], [0, 5], "--", color=R); axs[0].set_xlabel("actual"); axs[0].set_ylabel("predicted"); axs[0].set_title(f"Predicted vs actual (test R² = {r2_score(y_te, pred):.3f})")
axs[1].scatter(pred, res, s=3, alpha=.3, color=I); axs[1].axhline(0, color=R); axs[1].set_xlabel("predicted"); axs[1].set_ylabel("residual"); axs[1].set_title("Residuals: note the cap at 5.0")
imp = pd.Series(pi.importances_mean, X_te.columns).sort_values()
axs[2].barh(imp.index, imp.values, color=A); axs[2].set_title("Permutation importance (Δ RMSE)"); axs[2].tick_params(axis="y", labelsize=8)
save(f, "proj_eval")
f, ax = plt.subplots(figsize=(6, 2.8)); v = [t.value for t in study.trials]
ax.plot(v, "o", color=S, ms=4, label="trial"); ax.plot(np.minimum.accumulate(v), color=T, lw=2, label="best so far"); ax.set_xlabel("trial"); ax.set_ylabel("CV RMSE"); ax.legend(); ax.set_title("Optuna TPE search history")
save(f, "proj_optuna")
# ENDHIDE
