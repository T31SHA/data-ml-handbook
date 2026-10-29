# Data Visualisation

tagline: Choose the chart from the question, encode data with the most accurate visual channel, and make the insight the title.

## Choosing the chart from the question

![](figures/vizkinds.svg)

| Question | Best chart | Avoid | Code |
|---|---|---|---|
| Distribution of one variable | histogram, KDE, ECDF, box/violin | pie, 3-D | `sns.histplot`, `sns.ecdfplot`, `sns.violinplot` |
| Compare distributions across groups | small multiples, ridgeline, boxplot | overlapping histograms > 3 | `sns.displot(col=…)` |
| Relationship (2 numerics) | scatter (+ alpha), hexbin for big n, regression line | line plot on unordered x | `sns.scatterplot`, `plt.hexbin` |
| Ranking / comparison | sorted horizontal bar, dot plot | unsorted bars, radar | `sns.barplot(order=…)` |
| Change over time | line, area (stacked for parts) | bars for > 20 periods | `df.plot.line()` |
| Part-to-whole | stacked bar, treemap, waffle | pie with > 4 slices | `df.plot.bar(stacked=True)` |
| Correlation structure | clustered heatmap | unsorted heatmap | `sns.clustermap(corr)` |
| Geography | choropleth (rates!), point map | choropleth of raw counts | `geopandas`, `plotly.express` |
| Model diagnostics | residual plot, calibration curve, PR/ROC, learning curve, SHAP | accuracy bar alone | `sklearn.metrics.*Display` |

**Precision of visual channels** (Cleveland & McGill, most to least accurate): position on a common scale → position on unaligned scales → length → angle/slope → area → volume → colour saturation/hue. This ranking is why bar and dot charts beat pies, and why bubble charts are hard to read precisely.

## Annotated example: small multiples with an insight title

@@code code/s03_viz.py@@
@@out code/out/s03_viz.txt@@

![](figures/viz_demo.svg)

## Summary statistics lie without a picture

![](figures/correlation.svg)

!!! pitfall "Pitfall: trusting r without plotting"
    Pearson r only measures *linear* association and is dominated by outliers. A perfect parabola can have r ≈ 0, and a single outlier can manufacture a strong correlation (Anscombe's quartet, the Datasaurus dozen). Always plot.

## Design rules

| Rule | Practice |
|---|---|
| One message per chart | the title states the insight ("Churn doubles after three support calls"), not the variables |
| Honest axes | bars start at zero; log scale for multiplicative data, labelled as such; same y-scale across small multiples |
| Reduce ink | remove chart junk, heavy gridlines, 3-D, shadows; direct-label lines instead of legends |
| Colour with purpose | sequential (viridis) for magnitude, diverging (RdBu) around a meaningful midpoint, qualitative for categories (≤ 7); colour-blind safe |
| Show uncertainty | confidence bands, error bars, intervals, bootstrap distributions |
| Order matters | sort categories by value; chronological for time |
| Overplotting | alpha, hexbin, 2-D density, sampling |
| Accessibility | ≥ 8 pt text, don't encode only by red/green, add patterns or labels |

!!! tip "Pro tip: the four-plot model diagnostic"
    For any regression model, plot (1) predicted vs actual, (2) residuals vs predicted, (3) residual distribution, (4) residuals vs the top feature. For classifiers: confusion matrix at the chosen threshold, PR curve, calibration curve and score distributions per class. Section 03 shows the regression version.

| Tool | Best for |
|---|---|
| matplotlib | full control, publication figures, anything custom |
| seaborn | statistical plots from DataFrames in one line |
| plotly / altair | interactive exploration, web embedding, declarative grammar |
| Streamlit / Gradio / Dash | data apps and ML demos |
| Power BI / Tableau / Looker / Superset / Metabase | business dashboards on governed data |
| Grafana | operational metrics, model monitoring dashboards |
