"""
NYC Yellow Taxi Trip Analysis - Delivery Route Assistant
Main execution script that orchestrates the complete analysis pipeline
"""

import warnings
warnings.filterwarnings('ignore')

# Import modules from the new structure
from src.modules.data_loader import load_and_preprocess_data, save_results
from src.modules.exploratory_analysis import (
    perform_eda, 
    advanced_time_series_analysis, 
    payment_analysis, 
    create_dashboard_visualizations, 
    generate_report_summary
)
from src.modules.machine_learning import build_prediction_model, enhanced_model_analysis
from src.modules.route_optimization import route_optimization_analysis, geospatial_analysis
from src.modules.genai_integration import generate_technical_summaries, create_ai_analysis_report

def main():
    """
    Main function to execute complete analysis
    """
    print("NYC TAXI DATA ANALYSIS - DELIVERY ROUTE ASSISTANT")
    print("=" * 65)
    
    # Configuration
    DATA_FILE = "data/Cognizant assignment data - NYC Taxi 2023.csv"  # Updated path
    
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = load_and_preprocess_data(DATA_FILE)
        
        if df is None:
            print("Failed to load data. Please check file path and try again.")
            return
        
        # Core Analysis
        print("\nPerforming exploratory data analysis...")
        df_clean = perform_eda(df)
        
        # Advanced Analytics
        print("\nRunning advanced analytics...")
        advanced_time_series_analysis(df_clean)
        payment_analysis(df_clean)
        geospatial_analysis(df_clean)
        
        # Machine Learning
        print("\nBuilding prediction models...")
        model_results = build_prediction_model(df_clean)
        enhanced_models = enhanced_model_analysis(df_clean)
        
        # AI Integration
        print("\nGenerating AI-powered insights...")
        generate_technical_summaries()
        ai_report = create_ai_analysis_report(df_clean, model_results)
        
        # Route Optimization
        print("\nPerforming route optimization analysis...")
        route_optimization_analysis(df_clean, sample_size=15)
        
        # Dashboard
        print("\nCreating dashboard visualizations...")
        create_dashboard_visualizations(df_clean)
        
        # Executive Summary
        print("\nGenerating executive summary...")
        generate_report_summary(df_clean)
        
        # Save results
        print("\nSaving analysis results...")
        save_results(df_clean, model_results)
        
        print("\nANALYSIS COMPLETED SUCCESSFULLY!")
        print("\nFiles generated:")
        print("- outputs/taxi_analysis_cleaned_data.csv")
        print("- outputs/taxi_analysis_model.pkl")
        print("- outputs/nyc_taxi_pickup_map.html (if coordinates available)")
        print("- outputs/ (various visualization files)")
        
        return df_clean, model_results, ai_report
        
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")
        print("Please check your data file and dependencies.")
        return None, None, None
        
    finally:
        print("\nAnalysis script completed.")

def run_quick_analysis():
    """
    Run a quick analysis with basic insights
    """
    print("QUICK ANALYSIS MODE")
    print("=" * 30)
    
    DATA_FILE = "data/Cognizant assignment data - NYC Taxi 2023.csv"
    
    try:
        df = load_and_preprocess_data(DATA_FILE)
        if df is not None:
            perform_eda(df)
            generate_report_summary(df)
        else:
            print("Failed to load data for quick analysis.")
    except Exception as e:
        print(f"Quick analysis failed: {str(e)}")

def run_full_analysis():
    """
    Run complete analysis pipeline
    """
    return main()

if __name__ == "__main__":
    main()