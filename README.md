# Culinary Compass

## Overview

Culinary Compass is an AI-driven project designed to assist in identifying optimal locations for new restaurants using spatial data analysis and machine learning. This solution leverages historical health inspection data from New York City to provide actionable insights and interactive recommendations.

## Features

- **Interactive Map**: Displays restaurant recommendations with detailed ratings.
- **Custom Rating System**: Maps inspection grades and scores to an intuitive star-rating system.
- **Machine Learning Model**: Predicts restaurant success likelihood based on historical data.
- **Visual Insights**: Offers static and interactive visualizations for better decision-making.

## Technologies

- **Programming Languages**: Python (Pandas, NumPy, Scikit-learn, Folium)
- **Libraries**: Matplotlib, Seaborn
- **Visualization Tools**: Folium for interactive maps, static charts with Matplotlib and Seaborn
- **Development Tools**: Jupyter Notebook, Visual Studio Code

## Dataset

- **Source**: NYC Restaurant Inspection Results (2023) by the New York City Department of Health.
  **Link**:https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j
- **Key Fields**:
  - Restaurant Name (DBA)
  - ZIP Code
  - Health Inspection Grade
  - Numerical Inspection Score
  - Latitude/Longitude

## Data Processing

1. **Cleaning**: Removed duplicates and standardized ZIP codes.
2. **Handling Missing Data**: Addressed gaps in key fields like `GRADE` and `SCORE`.
3. **Feature Engineering**: Created a "Star Rating" system for intuitive grading.

## Methodology

### Preprocessing

- Addressed inconsistencies in ZIP codes and handled missing data.
- Encoded categorical variables.

### Machine Learning

- Applied a `RandomForestClassifier` model.
- Tuned hyperparameters using `GridSearchCV`.
- Achieved 82% accuracy on test data.

### Visualization

- **Static Charts**: Provided insights into grades and scores by ZIP code.
- **Interactive Map**: Visualized top-rated restaurants with detailed markers.

## Results

- **Model Accuracy**: Achieved 82% classification accuracy.
- **Recommendations**: Interactive map highlights optimal restaurant locations based on health ratings.

### Insights

- Business districts often cluster higher-rated restaurants.
- High traffic areas correlate with better inspection results.

## Future Work

- Expand to additional cities for broader applicability.
- Incorporate advanced machine learning models like XGBoost.
- Analyze cuisine-level trends to refine recommendations.
- Develop a mobile app for wider accessibility.

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alifiyah29/Culinary-Compass.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Culinary-Compass
   ```
3. Install required libraries:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn folium
   ```

### Running the Project

1. Run the script:
   ```bash
   python Culinary_Compass.py
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Launch the interactive map:
   ```bash
   python display_map.py
   ```

## Output

- **Static Charts**: Visualize star rating distribution and model accuracy.
- **Interactive Map**: Displays top-rated restaurants with detailed markers.

## Acknowledgments

Special thanks to Prof. Tam McCreless for project guidance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
