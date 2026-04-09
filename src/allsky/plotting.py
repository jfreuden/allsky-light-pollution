import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# Plotting Helper functions
def _weighted_fit_line(x, y, yerr):
    """
    Return a weighted linear fit and confidence band when possible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    x = x[valid]
    y = y[valid]
    yerr = yerr[valid]

    if len(x) == 0:
        return None

    if len(x) == 1 or np.unique(x).size == 1:
        beta = np.array([0.0, y[0]], dtype=float)
        x_grid = np.linspace(x[0] - 1.0, x[0] + 1.0, 2)
        y_grid = np.full_like(x_grid, y[0], dtype=float)
        return {
            "x": x,
            "y": y,
            "yerr": yerr,
            "beta": beta,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "y_lower": y_grid,
            "y_upper": y_grid,
            "m": beta[0],
            "b": beta[1],
        }

    w = 1.0 / np.square(yerr)
    X = np.column_stack([x, np.ones_like(x)])

    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    m, b = beta

    x_grid = np.linspace(x.min(), x.max(), 400)
    X_grid = np.column_stack([x_grid, np.ones_like(x_grid)])
    y_grid = X_grid @ beta

    y_fit_obs = X @ beta
    dof = max(len(x) - 2, 1)
    s2 = np.sum(w * (y - y_fit_obs) ** 2) / dof

    XtWX = X.T @ (w[:, None] * X)
    cov_beta = s2 * np.linalg.pinv(XtWX)

    y_var_grid = np.sum((X_grid @ cov_beta) * X_grid, axis=1)
    y_se_grid = np.sqrt(np.maximum(y_var_grid, 0.0))

    z = 1.96
    y_lower = y_grid - z * y_se_grid
    y_upper = y_grid + z * y_se_grid

    return {
        "x": x,
        "y": y,
        "yerr": yerr,
        "beta": beta,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "y_lower": y_lower,
        "y_upper": y_upper,
        "m": m,
        "b": b,
    }

alias_labels = {
    "D": "Nightly ",
    "W": "Weekly ",
    "M": "Monthly ",
    "Q": "Quarterly ",
    "A": "Yearly ",
    "H": "Hourly ",
    "T": "",
    "min": ""
}

def plot_brightness(input_df, title_suffix=None, period="D", ax=None, plot_best_fit=True):
    """Plot the brightness of the dataset. The "brightness" is defined as the average of the pixel values of the masked input image."""
    df = input_df[["timestamp", "image_mean", "image_std", "exposure"]].copy()

    plot_df = df.groupby(df["timestamp"].dt.to_period(period)).agg(
        mean_image_mean=("image_mean", "mean"),
        n=("image_mean", "size"),
        pooled_image_std=("image_std", lambda s: np.sqrt(np.mean(s ** 2))),
    ).reset_index()

    plot_df["date"] = plot_df["timestamp"].dt.to_timestamp()
    plot_df["error"] = plot_df["pooled_image_std"] / np.sqrt(plot_df["n"])
    plot_df = plot_df.drop(columns=["timestamp"]).dropna(subset=["date", "mean_image_mean", "error"]).sort_values("date")

    x = mdates.date2num(np.array(plot_df["date"]))
    y = plot_df["mean_image_mean"].to_numpy(dtype=float)
    yerr = plot_df["error"].to_numpy(dtype=float)

    fit = _weighted_fit_line(x, y, yerr)
    if fit is None:
        print(f"No valid data to plot for {title_suffix or 'this dataset'}.")
        return

    dates_grid = mdates.num2date(fit["x_grid"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.errorbar(
        plot_df["date"], y, yerr=yerr,
        fmt="o", markersize=3, capsize=3, alpha=0.5,
        label="Daily mean ± error",
    )

    if plot_best_fit:
        ax.plot(
            dates_grid, fit["y_grid"],
            color="red", linewidth=2,
            label=("Constant fit" if len(fit["x"]) == 1 else f"Weighted fit: y = {fit['m']:.4g}x + {fit['b']:.4g}"),
        )
        if len(fit["x"]) > 1:
            ax.fill_between(
                dates_grid, fit["y_lower"], fit["y_upper"],
                color="red", alpha=0.2, label="95% confidence band",
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Image Brightness")
    suffix = "" if title_suffix == "" or title_suffix is None else f" ({title_suffix})"
    ax.set_title(f"{alias_labels[period]}Image Brightness" + suffix)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_exposure(input_df, title_suffix=None, period="D", ax=None, plot_best_fit=True):
    """Plot the exposure time of the dataset."""
    df = input_df[["timestamp", "image_mean", "image_std", "exposure"]].copy()

    exposure_daily = df.groupby(df["timestamp"].dt.to_period(period)).agg(
        mean_exposure=("exposure", "mean"),
        n=("exposure", "size"),
        exposure_std=("exposure", "std"),
    ).reset_index()

    exposure_daily["date"] = exposure_daily["timestamp"].dt.to_timestamp()
    exposure_daily["error"] = exposure_daily["exposure_std"] / np.sqrt(exposure_daily["n"])
    exposure_daily = exposure_daily.drop(columns=["timestamp"]).dropna(subset=["date", "mean_exposure"]).sort_values("date")

    plot_df = exposure_daily.dropna(subset=["error"])
    x = mdates.date2num(np.array(plot_df["date"]))
    y = plot_df["mean_exposure"].to_numpy(dtype=float)
    yerr = plot_df["error"].to_numpy(dtype=float)

    fit = _weighted_fit_line(x, y, yerr)
    if fit is None:
        print(f"No valid data to plot for {title_suffix or 'this dataset'}.")
        return

    dates_grid = mdates.num2date(fit["x_grid"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.errorbar(
        plot_df["date"], y, yerr=yerr,
        fmt="o", markersize=3, capsize=3, alpha=0.6,
        label="Nightly mean exposure ± error",
    )

    if plot_best_fit:
        ax.plot(
            dates_grid, fit["y_grid"],
            color="red", linewidth=2,
            label=("Constant fit" if len(fit["x"]) == 1 else f"Weighted fit: y = {fit['m']:.4g}x + {fit['b']:.4g}"),
        )

        if len(fit["x"]) > 1:
            ax.fill_between(
                dates_grid, fit["y_lower"], fit["y_upper"],
                color="red", alpha=0.2, label="95% confidence band",
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Exposure Time")
    suffix = "" if title_suffix == "" or title_suffix is None else f" ({title_suffix})"
    ax.set_title(f"{alias_labels[period]}Mean Exposure Time" + suffix)
    ax.legend()
    fig.tight_layout()
    return ax


def plot_synthetic_luminous_flux(input_df, title_suffix=None, period="D", ax=None, plot_best_fit=True):
    """Plot the synthetic luminous flux of the dataset.
    The 'synthetic luminous flux' is defined as the average brightness per unit of exposure time.
    This enables comparing same-real-brightness images that are darker / brighter due to exposure time only."""
    df = input_df[["timestamp", "image_mean", "image_std", "exposure"]].copy()
    df = df.assign(luminous_flux=df["image_mean"] / df["exposure"])
    df = df.dropna(subset=["timestamp", "image_mean", "image_std", "exposure", "luminous_flux"])

    sigma_exposure = 0.0
    df["luminous_flux_error"] = np.sqrt(
        (df["image_std"] / df["exposure"])**2 +
        ((df["image_mean"] * sigma_exposure) / (df["exposure"]**2))**2
    )

    daily = df.groupby(df["timestamp"].dt.to_period(period)).agg(
        mean_luminous_flux=("luminous_flux", "mean"),
        n=("luminous_flux", "size"),
        flux_error_rss=("luminous_flux_error", lambda s: np.sqrt(np.sum(np.square(s))) / len(s)),
    ).reset_index()

    daily["date"] = daily["timestamp"].dt.to_timestamp()
    daily["error"] = daily["flux_error_rss"]
    daily = daily.drop(columns=["timestamp"]).dropna(subset=["date", "mean_luminous_flux"]).sort_values("date")

    plot_df = daily.dropna(subset=["error"])
    x = mdates.date2num(np.array(plot_df["date"]))
    y = plot_df["mean_luminous_flux"].to_numpy(dtype=float)
    yerr = plot_df["error"].to_numpy(dtype=float)

    fit = _weighted_fit_line(x, y, yerr)
    if fit is None:
        print(f"No valid data to plot for {title_suffix or 'this dataset'}.")
        return

    dates_grid = mdates.num2date(fit["x_grid"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.errorbar(
            plot_df["date"], y, yerr=yerr,
            fmt="o", markersize=3, capsize=3, alpha=0.6,
            label="Nightly mean luminous flux ± error",
    )

    if plot_best_fit:
        ax.plot(
            dates_grid, fit["y_grid"],
            color="red", linewidth=2,
            label=("Constant fit" if len(fit["x"]) == 1 else f"Weighted fit: y = {fit['m']:.4g}x + {fit['b']:.4g}"),
        )

        if len(fit["x"]) > 1:
            ax.fill_between(
                dates_grid, fit["y_lower"], fit["y_upper"],
                color="red", alpha=0.2, label="95% confidence band",
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean (synthetic) luminous flux")
    suffix = "" if title_suffix == "" or title_suffix is None else f" ({title_suffix})"
    ax.set_title(f"{alias_labels[period]} (Synthetic) Luminous Flux" + suffix)
    ax.legend()
    fig.tight_layout()
    return ax