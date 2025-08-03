# NYC Taxi Trip Analysis - Delivery Route Assistant

A comprehensive data analysis and machine learning project for NYC Yellow Taxi trip data, featuring exploratory data analysis, predictive modeling, route optimization, and AI-powered insights.

## Project Requirements & Implementation

### What Was Asked

The project requirements included:

1. **Modular Code Structure**: Refactor the monolithic `main.py` file into multiple, smaller, and more organized files to achieve better code structure and maintainability.

2. **Professional Code Style**: Remove all symbols and icons from the codebase to make it cleaner and more professional.

3. **Specific Data File Usage**: Use the specific data file "Cognizant assignment data - NYC Taxi 2023.csv" for all analysis.

4. **Complete File Organization**: Ensure all necessary files are properly organized in the folder structure.

### How It Was Implemented

#### 1. Modular Architecture Implementation
- **Before**: Single monolithic `main.py` file containing all functionality
- **After**: Organized into 5 specialized modules:
  - `src/modules/data_loader.py`: Data loading and preprocessing
  - `src/modules/exploratory_analysis.py`: EDA and visualizations  
  - `src/modules/machine_learning.py`: ML models and predictions
  - `src/modules/route_optimization.py`: Graph theory and optimization
  - `src/modules/genai_integration.py`: AI-powered analysis

#### 2. Professional Code Style Implementation
- **Removed**: All emoji icons (🚕, 📊, 🔍, ❌, 🎉, etc.)
- **Removed**: Bold formatting and special symbols
- **Replaced**: Bullet points (•) with standard dashes (-)
- **Result**: Clean, professional code appearance suitable for enterprise environments

#### 3. Data File Configuration
- **Specified**: Exact file name "Cognizant assignment data - NYC Taxi 2023.csv"
- **Located**: In `data/` directory for proper organization
- **Configured**: Automatic loading in `main.py` with correct path
- **Verified**: File presence and accessibility

#### 4. Complete File Structure
- **Created**: Professional Python package structure with `__init__.py` files
- **Organized**: Source code in `src/modules/` directory
- **Separated**: Configuration files (requirements.txt, README.md) in root
- **Established**: Output directory for generated files
- **Documented**: Comprehensive README with usage instructions

### Key Benefits Achieved

1. **Maintainability**: Each module has single responsibility
2. **Scalability**: Easy to add new features or modify existing ones
3. **Professionalism**: Industry-standard code organization
4. **Reusability**: Modules can be imported independently
5. **Clarity**: Clear separation of concerns and functionality

## Project Structure

```
cognizant/
├── main.py                          # Main execution script
├── requirements.txt                  # Python dependencies
├── README.md                        # Project documentation
├── data/                            # Data files
│   └── Cognizant assignment data - NYC Taxi 2023.csv
├── src/                             # Source code
│   ├── __init__.py
│   ├── modules/                     # Analysis modules
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading and preprocessing
│   │   ├── exploratory_analysis.py  # EDA and visualizations
│   │   ├── machine_learning.py     # ML models and predictions
│   │   ├── route_optimization.py   # Graph theory and route optimization
│   │   └── genai_integration.py    # AI-powered analysis
│   └── utils/                       # Utility functions
│       └── __init__.py
├── outputs/                         # Generated files
│   ├── taxi_analysis_cleaned_data.csv
│   ├── taxi_analysis_model.pkl
│   ├── nyc_taxi_pickup_map.html
│   └── visualization files (.png)
└── docs/                            # Documentation
```

## Features

### 1. Exploratory Data Analysis (EDA)
- Trip duration distribution analysis
- Revenue patterns by hour and day
- Pickup location distribution
- Passenger-tip relationship analysis
- Payment method analysis
- Time series analysis for demand patterns

### 2. Machine Learning Models
- Trip duration prediction using multiple algorithms
- Feature importance analysis
- Model performance comparison
- Cross-validation and hyperparameter tuning
- Enhanced model analysis with engineered features

### 3. Route Optimization
- Graph-based network analysis
- Shortest path algorithms
- Traveling Salesman Problem (TSP) approximation
- Delivery route optimization
- Network visualization

### 4. Geospatial Analysis
- Interactive mapping with Folium
- Pickup location visualization
- Coordinate-based analysis (when available)

### 5. AI-Powered Insights
- Technical summary generation
- Business recommendation engine
- Risk assessment and implementation roadmap
- Executive summary reports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cognizant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the data file is in the correct location:
```
data/Cognizant assignment data - NYC Taxi 2023.csv
```

## Usage

### Method 1: Direct Execution (Recommended)
```bash
# Navigate to project directory
cd cognizant

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the complete analysis
python main.py
```

### Method 2: Python Import
```python
# Import and run quick analysis
from main import run_quick_analysis
run_quick_analysis()

# Import and run full analysis pipeline
from main import run_full_analysis
df, model_results, ai_report = run_full_analysis()

# Import and run main function directly
from main import main
df, model_results, ai_report = main()
```

### Method 3: Step-by-Step Execution
```python
# Import specific modules
from src.modules.data_loader import load_and_preprocess_data
from src.modules.exploratory_analysis import perform_eda
from src.modules.machine_learning import build_prediction_model
from src.modules.route_optimization import route_optimization_analysis
from src.modules.genai_integration import generate_technical_summaries

# Load data
df = load_and_preprocess_data("data/Cognizant assignment data - NYC Taxi 2023.csv")

# Run individual analyses
df_clean = perform_eda(df)
model_results = build_prediction_model(df_clean)
route_optimization_analysis(df_clean)
generate_technical_summaries()
```

### Expected Output
When you run the analysis, you'll see:
- Data loading and preprocessing progress
- Exploratory data analysis results
- Machine learning model training and evaluation
- Route optimization analysis
- AI-powered insights generation
- Visualization files saved to `outputs/` directory

### Generated Files
After running the analysis, check the `outputs/` directory for:
- `taxi_analysis_cleaned_data.csv` - Preprocessed dataset
- `taxi_analysis_model.pkl` - Trained machine learning model
- Various `.png` files - Visualizations and charts
- `nyc_taxi_pickup_map.html` - Interactive map (if coordinates available)

## Key Deliverables

### Analysis Results
- **Cleaned Dataset**: Preprocessed taxi trip data
- **Trained Models**: Machine learning models for trip duration prediction
- **Visualizations**: Comprehensive charts and graphs
- **Interactive Map**: Geospatial visualization of pickup locations

### Business Insights
- Peak revenue hours and demand patterns
- Optimal fleet allocation strategies
- Route optimization recommendations
- Dynamic pricing opportunities
- Operational efficiency improvements

### Technical Reports
- Model performance metrics
- Feature importance rankings
- Prediction accuracy analysis
- Risk assessment and mitigation strategies

## Business Applications

### Fleet Management
- Optimize driver allocation based on demand patterns
- Reduce idle time through predictive positioning
- Improve customer satisfaction with faster pickup times

### Revenue Optimization
- Implement dynamic pricing during peak hours
- Focus resources on high-revenue locations
- Maximize fare collection through strategic routing

### Delivery Services
- Optimize delivery routes for corporate clients
- Reduce delivery time through intelligent routing
- Scale operations efficiently with predictive analytics

### Operational Efficiency
- Reduce fuel costs through optimized routes
- Improve driver utilization rates
- Enhance customer experience with accurate ETAs

## Customization

### Adding New Features
1. Create new functions in appropriate modules
2. Update imports in main.py
3. Add function calls to the main pipeline

### Modifying Analysis Parameters
- Adjust sample sizes in route optimization
- Change model hyperparameters in machine_learning.py
- Modify visualization settings in exploratory_analysis.py

### Extending AI Integration
- Uncomment OpenAI imports in genai_integration.py
- Add your API key
- Customize prompts for specific business needs

## Output Files

### Data Files
- `outputs/taxi_analysis_cleaned_data.csv`: Preprocessed dataset
- `outputs/taxi_analysis_model.pkl`: Trained machine learning model

### Visualizations
- `outputs/trip_duration_distribution.png`: Duration analysis
- `outputs/revenue_patterns.png`: Revenue by hour
- `outputs/pickup_distribution.png`: Location analysis
- `outputs/passenger_tip_relationship.png`: Tip analysis
- `outputs/payment_analysis.png`: Payment methods
- `outputs/time_series_analysis.png`: Temporal patterns
- `outputs/feature_importance.png`: ML feature importance
- `outputs/network_visualization.png`: Route network
- `outputs/dashboard.png`: Comprehensive dashboard

### Interactive Files
- `outputs/nyc_taxi_pickup_map.html`: Interactive map (if coordinates available)

## Technical Requirements

### Python Version
- Python 3.8 or higher

### Key Libraries
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, joblib
- **Graph Theory**: networkx, scipy
- **Geospatial**: folium
- **Optional AI**: openai, langchain

### System Requirements
- Minimum 4GB RAM
- 2GB free disk space for outputs
- Internet connection for map generation

## Troubleshooting

### Common Issues

1. **Data File Not Found**
   - Ensure the CSV file is in the `data/` directory
   - Check file name matches exactly: "Cognizant assignment data - NYC Taxi 2023.csv"

2. **Missing Dependencies**
   - Run: `pip install -r requirements.txt`
   - For specific errors, install missing packages individually

3. **Memory Issues**
   - Reduce sample size in route optimization
   - Use smaller dataset for testing
   - Close other applications to free memory

4. **Visualization Errors**
   - Ensure matplotlib backend is properly configured
   - Check if display is available (for headless servers)

### Performance Optimization

1. **Large Datasets**
   - Use data sampling for initial testing
   - Implement chunked processing for memory efficiency
   - Consider using dask for very large datasets

2. **Model Training**
   - Reduce hyperparameter search space
   - Use fewer estimators for faster training
   - Enable parallel processing where available

## API Integration

### OpenAI Integration
To enable AI-powered analysis:

1. Install OpenAI: `pip install openai`
2. Set your API key: `export OPENAI_API_KEY="your-key"`
3. Uncomment OpenAI imports in genai_integration.py
4. Replace sample outputs with actual API calls

### Custom API Integration
- Modify `create_ai_prompt_template()` for different AI services
- Update `generate_technical_summaries()` for custom prompts
- Extend `create_ai_analysis_report()` for specific business needs

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Make your changes
5. Add tests for new functionality
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to all functions
- Include type hints where appropriate

### Testing
- Add unit tests for new modules
- Test with different dataset sizes
- Verify output file generation
- Check visualization quality

## License

This project is for educational and research purposes. Please ensure compliance with data usage agreements when working with NYC taxi data.

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments and docstrings
3. Create an issue with detailed error information
4. Include system specifications and Python version

---

**Note**: This project uses the NYC Taxi trip data file "Cognizant assignment data - NYC Taxi 2023.csv" which is already included and configured in the main.py file. The analysis is designed to work with this specific dataset format and will automatically load it from the data/ directory. 