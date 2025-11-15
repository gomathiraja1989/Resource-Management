# Industrial Human Resource Geo-Visualization

## Problem Statement

In India, understanding the distribution of the workforce across various sectors is essential for effective resource management and policy-making. The existing classification of main and marginal workers by sector is outdated and does not reflect the current economic or employment trends. This project aims to update and analyze the industrial classification of main and marginal workers (excluding cultivators and agricultural laborers), broken down by sex, section, division, and class, to provide accurate data for policy and planning interventions.

This analysis leverages recent datasets covering the state-wise distribution of workforce among various industries, such as manufacturing, retail, construction, and more. The outcome is a geo-visual analytical dashboard enabling exploration of the workforce landscape across geographies and industries in India, supporting evidence-based decisions for public and private stakeholders.

---

## Approach

The project follows a modular, stepwise methodology to ensure maintainability and reproducibility:

1. **Data Loading & Merging**
   - Import all available CSV files containing industrial classification data.
   - Merge the files to form a master DataFrame.

2. **Exploratory Data Analysis (EDA)**
   - Assess missing data, outliers, and statistical summaries.
   - Visualize data distribution by industry, region, gender, and type (main/marginal).

3. **Data Cleaning**
   - Handle missing/ambiguous values and normalize categorical labels.
   - Remove duplicates or erroneous records.

4. **Feature Engineering**
   - Create/encode variables useful for downstream analysis (e.g., total workers, sector groupings).

5. **Natural Language Processing (NLP)**
   - Analyze industry descriptions and cluster/group similar business categories (Retail, Poultry, Manufacturing, etc.).
   - Use NLP to automate grouping in the absence of clean, structured classification.

6. **Machine Learning (Optional)**
   - Apply ML algorithms if predictive modeling or advanced analytics are suitable.

7. **Visualization & Dashboard**
   - Build a Streamlit app with interactive geo-visualizations (via plotly) to display workforce distributions.
   - Include filters for geography, gender, sector, type, and other attributes.

8. **Result Analysis & Reporting**
   - Summarize findings, actionable insights, and policy implications.
   - Prepare outputs for presentation and stakeholder communication.

---

## Project Workflow

The workflow for this project follows a logical progression:

```
Data Ingestion
   ↓
Data Merging & Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering/NLP Industry Grouping
   ↓
Visualization & Dashboard App (Streamlit + Plotly)
   ↓
Result Analysis, Reporting & Presentation
```

Alternatively, in stepwise markdown:
1. Data ingestion (CSV files)
2. Data merging/cleaning
3. EDA and visualization
4. Feature engineering, NLP groupings
5. Dashboard development (Streamlit)
6. Results analysis and presentation (PPT/demo video)

---

## Exploratory Data Analysis (EDA) Insights

The EDA phase investigates the data through the following key steps and visual summaries:

- **Missing Data Heatmap:** Identify missing, inconsistent, or anomalous records.
- **Distribution Plots:** State-wise and sector-wise worker counts.
- **Gender and Worker Type Breakdown:** Main vs. marginal, male vs. female population analysis by sector/region.
- **Industry Trends:** Highlight sectors with the highest employment, emerging trends, and regional disparities.
- **Geographical Visualization:** Choropleth and map plots to display workforce distribution across states/union territories.
- **Business Category Groupings:** NLP-driven groupings of industry descriptions and their headcounts.

Sample figures and their insights will be included once EDA is completed. These analytics lay the foundation for meaningful dashboard visualizations and actionable conclusions.

---

## Dashboard Visualization

The project's results are delivered through an interactive dashboard built with Streamlit and Plotly, allowing dynamic exploration of industrial workforce data.

**Key Features:**
- **Geo-Visualization:** State-wise maps and choropleth plots of workforce distribution.
- **Industry-Sector Filtering:** Drill down by business sector, division, and category (via grouped NLP classes).
- **Demographic Filtering:** Toggle between main/marginal, male/female data views.
- **Data Summary:** Tabular and graphical summaries by region and industry.
- **Interactive Elements:** Selectors, sliders, and tooltips for granular exploration.

**Usage:**
- Launch the app with `streamlit run app/dashboard.py` (see full usage below).
- Adjust filters in the sidebar to focus on specific sectors, regions, or worker categories.


## Demo and Presentation

- [Demo Video](#) _(Link will be updated post-project completion)_
- [PowerPoint Presentation](#) _(Link will be updated post-project completion)_
- [LinkedIn Post](#) _(Link will be updated after sharing the project)_

---

## Installation and Usage

The repository is designed for easy setup and local execution. Follow these steps to get started:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/industrial-human-resource-geo-visualization.git
cd industrial-human-resource-geo-visualization
```

### 2. Install Dependencies
Install all required libraries using pip (recommended to use a virtual environment):
```bash
pip install -r requirements.txt
```
_Alternatively, use `conda` and the provided `environment.yml` if available._

### 3. Prepare Data
- Place raw CSV dataset files in the `DataSets/` directory.

### 4. Run the Dashboard App
```bash
streamlit run app/dashboard.py
```
- The app will open in your default browser.
- Configure filters to explore workforce statistics interactively.

---

## Tools and Technologies Used

- **Programming Language:** Python 3.x
- **Data Manipulation & EDA:** pandas, numpy
- **Visualization:** plotly, matplotlib, seaborn
- **Dashboard:** streamlit
- **Machine Learning & Feature Engineering:** scikit-learn
- **Natural Language Processing:** nltk, spacy
- **Other Useful Libraries:** os, glob, re (as needed for data wrangling)
- **Version Control:** git, GitHub (public repository)

---

## Dataset

The dataset contains state-wise counts of industrial classification for main and marginal workers of males and females employed in various industrial activities (excluding cultivators and agricultural laborers) across India.

- **Key Features:**
  - State/Region
  - Category of worker (Main/Marginal, Male/Female)
  - Industry/Section/Division/Class (e.g., manufacturing, construction, retail, furniture, plastics, chemicals, etc.)
  - Worker population counts per industry sector per geographic region