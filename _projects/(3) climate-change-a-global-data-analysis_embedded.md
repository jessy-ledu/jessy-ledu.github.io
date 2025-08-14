---
name: "Climate Change, Emissions & Demography"
tools: [Python, Pandas, Plotly, Quarto]
image: "https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change.png"
description: "Global analysis of warming, emissions, and population using Python."
toc: true
toc-title: "Table of Contents"
---

# Global analysis of warming, emissions, and population using Python.
## Introduction
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

I explore how countries vary in:
- Climate change trends (°C/decade)
- Greenhouse gas (GHG) emissions
- Population growth

Through this, I aim to uncover global patterns and inequalities in **climate impact**, **contribution**, and **demographic change**, using accessible modeling techniques such as **linear regression** and **PCA**.

---

This notebook not only addresses a critical global issue, but also serves as a practical demonstration of **end-to-end data analysis in Python** for potential collaborators or employers.
For those who want to explore the **full notebook and code**, you can jump directly to the [notebook section](#full-notebook).

> **Disclaimer**: This notebook is intended as a **data science portfolio project**, not a scientific publication.  
> It uses publicly available data and simplified models to explore global patterns in climate, emissions, and population.  
> While care was taken to ensure accuracy, the analysis is exploratory in nature and meant primarily to demonstrate **technical and analytical skills** in Python.

---

## Mean Global Surface Temperature Change
This plot shows the global average change in surface temperature over recent decades, with the option to view individual countries using the dropdown menu. It provides a clear view of the overall warming trend while allowing for country-level comparisons. Serving as a visual starting point for exploring climate patterns, it highlights both the magnitude and pace of temperature change, laying the groundwork for deeper analyses of the factors driving these shifts and their potential impacts.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Mean Global Surface Temperature Change
</div>
<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/interactive_plot_Global_Surface_temp.html" 
        width="100%" 
        height="600" 
        style="border:none;">
</iframe>

###  Linear Temperature Trends by Country (°C/decade)
Since a linear model can effectively approximate the observed trends at both global and local scales, I applied simple linear regression to each country’s time series data to quantify changes in climate indicators such as surface temperature. The slope of the fitted line represents the average rate of temperature change per year, which I then scaled to °C per decade for more straightforward interpretation and comparison across regions.

### Why Linear Regression?

Linear regression is a straightforward method to model long-term trends, offering a **first-order estimate** of how an indicator changes over time. It captures:

-  **The direction** of change (warming or cooling)
-  **The rate** of change (slope in °C/year → °C/decade)
-  While it doesn't capture non-linear effects or fluctuations, it's widely used as a baseline trend indicator.

### Country Code Reference

The table below lists the ISO 3166-1 alpha-3 (ISO3) codes for the countries used in this study.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
List of ISO 3 codes
</div>
<iframe
  src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/table_countries-names.html"
  width="100%"
  height="520"
  style="border:none;"
  loading="lazy"
></iframe>

### 🔺 Top 5 Countries with the **Largest Warming Trends** (°C/decade)

| Country | Trend (°C/decade) |
|---------|------------------:|
| Ukraine (UKR) | **0.764** |
| Moldova (MDA) | 0.750 |
| Azerbaijan (AZE) | 0.674 |
| Georgia (GEO) | 0.662 |
| Armenia (ARM) | 0.651 |

These countries are experiencing the **fastest warming**, with Ukraine and Moldova showing particularly steep trends — more than **seven times** the global average rate.

---

### 🔻 Top 5 Countries with the **Smallest Positive Warming Trends** (°C/decade)

| Country | Trend (°C/decade) |
|---------|------------------:|
| Chile (CHL) | 0.085 |
| Yemen (YEM) | 0.116 |
| French Polynesia (PYF) | 0.129 |
| Timor-Leste (TLS) | 0.131 |
| Argentina (ARG) | 0.134 |

These countries are still warming, but at a **much slower pace**. Regional climate influences, or oceanic buffering, may contribute to these lower rates.

---

### No Cooling Countries Identified

> **No countries show a stable or cooling trend (≤ 0 °C/decade).**

According to the 1961–2022 linear trends, **every country with sufficient data** is warming. This aligns with the broader global warming consensus.

---
<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Surface Temperature per Country Colored by Change Magnitude
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/Surface_Temperature.png" 
     alt="Global Climate Analysis" 
     width="100%" 
     style="border:0;">

### Conclusion and Final Map

Linear-trend analysis provides a clear, comparable metric for warming rates across countries. Most nations with available data show increasing surface temperatures—a hallmark of global warming—though the pace varies substantially by region.

The map below displays this distribution; the color gradient indicates the magnitude of change (°C per decade).

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Rate of Temperature Change by Country
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/Rate_Temperature_Change.png" 
     alt="Map of country-level warming rates (°C per decade)" 
     width="100%" 
     style="border:0;">

## Global Warming Trends Visualized on the World Map

To complement our quantitative analysis, I mapped the **linear temperature trends** by country across the globe. Each country is colored based on its **warming rate (°C/decade)**, derived from simple linear regression on its historical temperature data (1961–2022).

###  What Does the Map Show?

- **All countries** with sufficient data show a **positive trend**, confirming that **no region is cooling** over the long term.
- **Darker shades** represent countries experiencing **slower warming**.

### Regional Highlights

- **Northern countries**—including **Russia**, **Eastern Europe**, and **Canada**—stand out with **higher warming trends**. This reflects a well-known climate phenomenon:  
  >  **Polar and sub-polar amplification**, where higher latitudes warm faster due to snow/ice feedbacks and changes in atmospheric circulation.

- **Equatorial and some Southern Hemisphere countries** show **weaker warming trends**, though still positive. These include parts of **South America**, **Southeast Asia**, and **Oceania**.

###  Interpretation Notes

- This map provides an intuitive, spatial view of how climate change is **unevenly distributed**, despite being **global phenomenon**.
- Geographic patterns help identify **climate hotspots**, support **policy targeting**, and inspire **region-specific climate adaptation strategies**.

---
## Factors influencing global warming- studying impacting factors at global and local scales

To explain the observed global trend—or at least to identify variables that correlate with it—I will analyze additional demographic indicators alongside greenhouse gas (GHG) emissions, assessing patterns at both the global and country-specific levels.

---
### Average Population Growth Rate (% per year)

I compute the **average annual population growth rate** using:

**Average Annual Growth Rate (% per year)** is computed as:  
$ \text{Growth Rate (\%/year)} = \frac{\ln(P_{\text{end}}) - \ln(P_{\text{start}})}{\text{years}} \times 100 $


This assumes **exponential growth** and enables fair comparison across countries.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Population Growth Rate by Country (1960-2024)
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/Population_Growth_Rate.png" 
     alt="Global Climate Analysis" 
     width="100%" 
     style="border:0;">
     
### What the Map Shows

- **Fastest-growing populations** are mostly in **Sub-Saharan Africa** (e.g., Niger, Angola, Chad) and parts of **Central Asia**.
- **Slow or negative growth** is seen in **Russia**, **Western Europe** (e.g., Italy, Germany), and **some Eastern European** countries.
- **Eastern Europe** shows mixed patterns: some countries (e.g., Bulgaria) are shrinking, while others (e.g., Uzbekistan) grow faster.

These trends reflect a complex mix of **birth rates, aging, migration**, and **economic conditions** across regions.

---

### Greenhouse Gases (GHG) and Their Impact on Climate

The following map visualizes **linear trends in greenhouse gas emissions** for each country. The trends are estimated using the same **simple linear regression approach** applied previously to temperature changes, allowing for consistent comparison across indicators.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Greenhouse Gases Eissions Trend per Country
</div>
<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/interactive_map_with_dropdown_GHG_trend.html" 
        width="100%" 
        height="600" 
        style="border:none;">
</iframe>

### Interpreting the Map

- **Carbon dioxide (CO₂)** emissions generally show an upward trend in countries such as **China, India, and the United States**, while parts of **Western Europe** display stable or slightly declining trends.  
- **Fluorinated gases (F-gases)** exhibit largely stagnant emissions globally, with increases mostly concentrated in countries with high CO₂ emissions.  
- **Methane (CH₄)** emissions present contrasting patterns: **Northern countries** (e.g., Canada, Northern Europe) often show decreasing trends, whereas **Southern and densely populated countries** like China, India, and Brazil display increasing trends.  
- **Nitrous oxide (N₂O)** emissions are increasing in countries including China, India, Brazil, and the United States, while trends are declining or stable in **Northern and Eastern European countries** and Russia.

Overall, total GHG trends reflect a **divergence between countries**: some are increasing emissions by up to **+200 Mt CO₂e per decade**, while others are decreasing by similar magnitudes.

The following map shows the **cumulative GHG emissions for each country** from **1970 to 2023**, providing insight into long-term contributions:

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Cumulative Greenhouse Gases Emissions per Country
</div>
<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/interactive_map_with_dropdown_GHG_cum.html" 
        width="100%" 
        height="600" 
        style="border:none;">
</iframe>

### Interpreting Cumulative Emissions

Cumulative emissions highlight that countries with **declining current trends** can still be among the **largest historical emitters**. For example:

- **Europe and Russia**: Emissions are decreasing now, but have historically contributed a significant share.  
- **United States and China**: These countries remain among the **highest emitters historically and currently**, maintaining substantial emissions.  
- **Africa, Latin America, and the Indo-Pacific region**: Cumulative contributions remain relatively small on a global scale.

Note that Alaska is grouped with the United States in this analysis, so it shares the same color on the map. While permafrost thawing in Alaska does release some greenhouse gases, primarily methane, its contribution is small compared to the overall emissions from the United States and other major industrial sources.

---

## Exploring Correlations Between Climate, Population, and Emissions

To better understand the **drivers and patterns of climate change**, I analyzed the **average values per country** for:

- **Surface temperature change (°C/decade)**
- **Population growth**
- **Greenhouse gas (GHG) emissions**

### Correlation Analysis

The following plots compare country-level averages (x vs. y) and overlay five fitted models in each case: **linear**, **polynomial (degree 2 & 3)**, **logarithmic**, and **exponential**. Across all three relationships, the models fit the data **consistently well**, indicating a strong, stable association that is already well captured by a simple linear baseline.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Model Comparison: Climate Indicator vs Mean GHG
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/Climate_Indicator_vs_Mean_GHG.png" 
     alt="Climate Indicator vs Mean GHG with multiple model fits" 
     width="100%" 
     style="border:0;">

**What this shows.** Countries with higher average GHG emissions tend to exhibit stronger climate indicator values (positive association). All five models fit closely, with **linear** performing on par with the more flexible forms—suggesting limited gains from added complexity for this relationship.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Model Comparison: Climate Indicator vs Population
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/Climate_Indicator_vs_pop.png" 
     alt="Climate Indicator vs Population with multiple model fits" 
     width="100%" 
     style="border:0;">

**What this shows.** The climate indicator increases with population scale. All models (linear, poly-2/3, log, exponential) **fit the data well**, again indicating that a **linear fit is a strong, interpretable baseline** for cross-country comparisons.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
Model Comparison: Mean Population vs Mean GHG
</div>
<img src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/GHG_vs_pop.png" 
     alt="Mean Population vs Mean GHG with multiple model fits" 
     width="100%" 
     style="border:0;">

**What this shows.** More populous countries generally emit more GHGs. All five models achieve **similarly strong fits**, with higher-order curves offering only marginal visual improvement over the **linear** specification.

> **Takeaway.** Because all models fit well and patterns are monotonic, a **linear model** is adequate for summary and communication, with polynomial/log/exponential alternatives providing only incremental flexibility.


### Observed Patterns

- **GHG emissions and temperature change** are positively correlated globally.  
- **Population and GHG emissions** also correlate positively, likely reflecting:
  - Higher energy demand in more populous countries.
  - Increased industrial and agricultural activity.  
- **Population and temperature trends** show an indirect correlation, potentially mediated by emissions and economic activity.

> While correlation does **not imply causation**, these relationships provide a **first-order understanding** of global climate dynamics.

### Model Fitting and Trend Behavior

I tested different model types — **linear**, **logarithmic**, and **exponential** — to capture these relationships. Performance was similar across models, supporting **linear regression** as:

- A **simple and interpretable approach**.
- Adequate for **cross-country comparisons**.
- Useful for **communicating trends** clearly in a portfolio context.

---

### Principal Component Analysis (PCA): Synthesizing Global Patterns

To summarize complex relationships between **climate indicators**, **GHG emissions**, and **population growth**, I performed a **Principal Component Analysis (PCA)**. PCA reduces dimensionality while preserving the most important patterns.

<div style="text-align:center; font-weight:bold; font-size:1.3em; margin-bottom:0.5em;">
    PCA Biplot — Country Profiles
</div>
<iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/PCA_plot_Country_profiles.html" 
        width="100%" 
        height="500" 
        style="border:none;">
</iframe>

#### Key Insights from PCA

- **Major GHG emitters** (e.g., **USA**, **China**) appear as outliers due to high emissions.  
- Excluding these outliers shows consistent underlying global trends.  
- **GHG emissions** exhibit **weak correlations** with population growth and temperature change, highlighting that emissions alone do not explain climate impacts fully.  
- An **inverse relationship** is observed between **temperature trend** and **population growth**:
  - Countries with strong warming trends often have slower population growth (e.g., **Zimbabwe**, **Moldova**).  
  - Countries with rapid population growth (e.g., **UAE**, **Qatar**) tend to have lower observed temperature trends.  

### A Global Process with Unequal Local Impacts

These results emphasize that while **climate change is global**, its **responsibility and impacts are unevenly distributed**:

- Some highly affected countries contribute little to emissions.  
- Major emitters may experience less immediate warming.  
- Geographic, economic, and infrastructural factors shape both vulnerability and contribution.

> This asymmetry supports the need for **differentiated responsibilities** and **tailored adaptation strategies** in climate policy.

### Integrating Correlations and PCA

By combining **correlation analysis** and **PCA**, I provide a coherent view of how **demographic**, **industrial**, and **climatic variables** interact at a global scale. The results underscore the complexity of global climate dynamics while remaining interpretable for **portfolio demonstration purposes**.

---

## Conclusion

This project confirms that **climate change is a global process**, with virtually every country experiencing some degree of warming over the past six decades. However, the **impact**, **contribution**, and **capacity to respond** vary widely across countries:

- **No country shows a cooling trend**, but some—such as **Zimbabwe (ZWE)** and **Moldova (MDA)**—are warming faster despite **low emissions** and **limited population growth**.  
- Major emitters—like **USA** and **China (CHN)**—have **substantially higher GHG outputs**, yet they are **not always the most climatically impacted**.  
- Countries with **rapid population or economic growth** (e.g., **UAE, ARE**; **Qatar, QAT**) often show **lower observed warming**, reflecting geographic factors, economic buffering, or infrastructure resilience.

Through **correlation analysis**, **linear regression models**, and **Principal Component Analysis (PCA)**, I identified key structural imbalances:  

> Countries experiencing the steepest climate impacts have **often contributed the least** to global emissions, highlighting a fundamental issue of **climate justice**.

---

### Skills Demonstrated

This project also illustrates **end-to-end data science capabilities**:

- **Data ingestion** from open-access sources (IMF, World Bank, etc.)  
- **Data cleaning & integration** across multiple country-level datasets  
- **Exploratory Data Analysis (EDA)** using statistical summaries, correlation metrics, and visualizations  
- **Geospatial visualization** of global trends with Matplotlib, Seaborn, and GeoPandas  
- **Regression modeling** (linear, logarithmic, exponential) to quantify temporal trends  
- **Dimensionality reduction** with PCA to reveal underlying patterns and relationships  
- **Narrative storytelling** with Markdown to communicate insights clearly and coherently  

---

> Thank you for exploring this analysis — feedback and discussion are welcome.  
>
> > **Note:** This project was developed with guidance from OpenAI's ChatGPT for code structuring, reasoning, and Markdown formatting.  
> All code, analyses, and visualizations were **independently validated and adapted** by me to ensure accuracy and reproducibility.

     
---
## Full Notebook
<a id="full-notebook"></a>

Below, you can view the entire notebook used to generate the visualizations and interpretations. This HTML document has been generated using Quarto from Python 3 code:

---
<div style="background-color:#1e1e1e; padding:10px; border-radius:5px;">
  <iframe src="https://jessy-ledu.github.io/assets/Projects/climate-change-a-global-data-analysis/climate-change-a-global-data-analysis_embedded.html" 
          width="100%" height="900" style="border:0; background-color:#1e1e1e;">
  </iframe>
</div>

