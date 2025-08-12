---
name: "Climate Change, Emissions & Demography"
tools: [Python, Pandas, Plotly, Quarto]
image: "https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change.png"
description: "Global analysis of warming, emissions, and population using Python."
---

### Project Context

This project is part of a **data science career-building portfolio**, designed to showcase the use of **Python for data analysis and visualization** in the context of real-world, global-scale issues.

The goal is to publish a well-documented, insightful notebook on Kaggle that demonstrates:
- Data cleaning and wrangling
- Exploratory data analysis (EDA)
- Regression and dimensionality reduction (PCA)
- Geographic visualization
- Interpretation and communication of findings

### Data Sources

This analysis uses **freely available, reputable datasets** from:
- [IMF Climate Change Indicators Dashboard](https://climatedata.imf.org/pages/climatechange-data)
- [World Bank Open Data](https://data.worldbank.org/)

### Objective

We explore how countries vary in:
- Climate change trends (°C/decade)
- Greenhouse gas (GHG) emissions
- Population growth

Through this, we aim to uncover global patterns and inequalities in **climate impact**, **contribution**, and **demographic change**, using accessible modeling techniques such as **linear regression** and **PCA**.

---

This notebook not only addresses a critical global issue, but also serves as a practical demonstration of **end-to-end data analysis in Python** for potential collaborators or employers.

> **Disclaimer**: This notebook is intended as a **data science portfolio project**, not a scientific publication.  
> It uses publicly available data and simplified models to explore global patterns in climate, emissions, and population.  
> While care was taken to ensure accuracy, the analysis is exploratory in nature and meant primarily to demonstrate **technical and analytical skills** in Python.

---

---
## Mean Global Surface Temperature Change
This plot shows the global average change in surface temperature over recent decades, with the option to view individual countries using the dropdown menu. It provides a clear view of the overall warming trend while allowing for country-level comparisons. Serving as a visual starting point for exploring climate patterns, it highlights both the magnitude and pace of temperature change, laying the groundwork for deeper analyses of the factors driving these shifts and their potential impacts.


<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/interactive_plot_Global_Surface_temp.html" 
        width="100%" 
        height="600" 
        style="border:none;">
</iframe>


##  Linear Temperature Trends by Country (°C/decade)
Since a linear model can effectively approximate the observed trends at both global and local scales, we applied simple linear regression to each country’s time series data to quantify changes in climate indicators such as surface temperature. The slope of the fitted line represents the average rate of temperature change per year, which we then scaled to °C per decade for more straightforward interpretation and comparison across regions.

### Why Linear Regression?

Linear regression is a straightforward method to model long-term trends, offering a **first-order estimate** of how an indicator changes over time. It captures:

-  **The direction** of change (warming or cooling)
-  **The rate** of change (slope in °C/year → °C/decade)
-  While it doesn't capture non-linear effects or fluctuations, it's widely used as a baseline trend indicator.

---

### 🔺 Top 5 Countries with the **Largest Warming Trends** (°C/decade)

| Country | Trend (°C/decade) |
|---------|------------------:|
| Zimbabwe (ZWE) | **0.148** |
| Ukraine (UKR) | 0.076 |
| Moldova (MDA) | 0.075 |
| Azerbaijan (AZE) | 0.067 |
| Georgia (GEO) | 0.066 |

These countries are experiencing the **fastest warming**, with Zimbabwe showing a particularly steep trend of nearly **0.15°C per decade**, well above the global average.

---

### 🔻 Top 5 Countries with the **Smallest Positive Warming Trends** (°C/decade)

| Country | Trend (°C/decade) |
|---------|------------------:|
| Chile (CHL) | 0.009 |
| Yemen (YEM) | 0.012 |
| French Polynesia (PYF) | 0.013 |
| Timor-Leste (TLS) | 0.013 |
| Argentina (ARG) | 0.013 |

These countries still show warming, but at a **much slower pace**, close to **0.01°C per decade**, which may reflect regional climatic factors, buffering effects, or measurement variability.

---

### No Cooling Countries Identified

** No countries show a stable or cooling trend (≤ 0 °C/decade).**

This suggests that, according to the linear trend from 1961–2022, **every country with sufficient data** is experiencing some level of warming. This aligns with the broader global warming consensus observed in both satellite and ground station data.

---
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change-a-global-data-analysis_embedded_files/figure-html/cell-11-output-1.png" 
     alt="Global Climate Analysis" 
     width="100%" 
     style="border:0;">



### Conclusion and Final Map

Linear-trend analysis provides a clear, comparable metric for warming rates across countries. Most nations with available data show increasing surface temperatures—a hallmark of global warming—though the pace varies substantially by region.

The map below displays this distribution; the color gradient indicates the magnitude of change (°C per decade).

<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change-a-global-data-analysis_embedded_files/figure-html/cell-14-output-1.png" 
     alt="Map of country-level warming rates (°C per decade)" 
     width="100%" 
     style="border:0;">



---

---

---
## Factors influencing global warming- studying impacting factors at global and local scales

In order to explain the observed global trend—or at least to identify variables that correlate with it—we will analyze additional demographic indicators alongside greenhouse gas (GHG) emissions, assessing patterns at both the global and country-specific levels.

---
## Average Population Growth Rate (% per year)

We compute the **average annual population growth rate** using:

**Average Annual Growth Rate (% per year)** is computed as:  
$ \text{Growth Rate (\%/year)} = \frac{\ln(P_{\text{end}}) - \ln(P_{\text{start}})}{\text{years}} \times 100 $


This assumes **exponential growth** and enables fair comparison across countries.

### What the Map Shows

- **Fastest-growing populations** are mostly in **Sub-Saharan Africa** (e.g., Niger, Angola, Chad) and parts of **Central Asia**.
- **Slow or negative growth** is seen in **Russia**, **Western Europe** (e.g., Italy, Germany), and **some Eastern European** countries.
- **Eastern Europe** shows mixed patterns: some countries (e.g., Bulgaria) are shrinking, while others (e.g., Uzbekistan) grow faster.

These trends reflect a complex mix of **birth rates, aging, migration**, and **economic conditions** across regions.
---

Below, you can view the entire notebook used to generate the visualizations and interpretations. This HTML document has been generated using Quarto:

---
<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change-a-global-data-analysis_embedded.html" width="100%" height="900" style="border:0;"></iframe>
