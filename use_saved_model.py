"""
Use Saved Sales Forecasting Model
Load the trained model and make predictions without retraining
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Import the forecaster class
from sales_forecast import CausalSalesForecast

def predict_future_sales(days_ahead=30):
    """
    Load model and predict future sales
    """
    print("="*60)
    print("LOADING SAVED MODEL")
    print("="*60)
    
    # Initialize forecaster
    forecaster = CausalSalesForecast()
    
    # Load the saved model
    if not forecaster.load_model('sales_forecast_model.pkl'):
        print("\nError: Could not load model. Train a model first by running:")
        print("  python sales_forecast.py")
        return None
    
    # Load your data to get the last known date and values
    try:
        df = forecaster.load_data('dummy_sales_data_2_years.csv')
    except Exception as e:
        print(f"\nError loading data: {e}")
        return None
    
    # Aggregate by date
    df_agg = df.groupby('date')['sales'].sum().reset_index()
    
    # Get last date
    last_date = df_agg['date'].max()
    print(f"\nLast known date: {last_date}")
    print(f"Predicting {days_ahead} days into the future...")
    
    # Create future dates
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=days_ahead,
        freq='D'
    )
    
    # Get last row of aggregated data as template
    last_row = df_agg.iloc[-1:].copy()
    future_data = pd.concat([last_row] * days_ahead, ignore_index=True)
    future_data['date'] = future_dates
    
    # Engineer features
    future_features = forecaster.engineer_features(future_data)
    
    # Ensure we have all the feature columns
    for col in forecaster.feature_columns:
        if col not in future_features.columns:
            future_features[col] = 0
    
    # Select only the features used in training
    X_future = future_features[forecaster.feature_columns]
    X_future = X_future.fillna(X_future.mean())
    
    # Scale features
    X_future_scaled = forecaster.scaler.transform(X_future)
    
    # Make predictions
    predictions = forecaster.model.predict(X_future_scaled)
    
    # Create results dataframe
    results = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predictions
    })
    
    # Display results
    print("\n" + "="*60)
    print("FUTURE SALES PREDICTIONS")
    print("="*60)
    print(results.to_string(index=False))
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Predicted Sales:    ${predictions.sum():,.2f}")
    print(f"Average Daily Sales:      ${predictions.mean():,.2f}")
    print(f"Median Daily Sales:       ${np.median(predictions):,.2f}")
    print(f"Min Daily Sales:          ${predictions.min():,.2f}")
    print(f"Max Daily Sales:          ${predictions.max():,.2f}")
    print("="*60)
    
    # Save to CSV
    results.to_csv('future_predictions.csv', index=False)
    print(f"\n✓ Predictions saved to: future_predictions.csv")
    
    return results


def predict_with_custom_scenarios():
    """
    Make predictions with custom scenario changes
    """
    print("="*60)
    print("SCENARIO-BASED PREDICTIONS")
    print("="*60)
    
    # Initialize and load model
    forecaster = CausalSalesForecast()
    if not forecaster.load_model('sales_forecast_model.pkl'):
        return None
    
    # Load data
    try:
        df = forecaster.load_data('dummy_sales_data_2_years.csv')
    except Exception as e:
        print(f"\nError loading data: {e}")
        return None
    
    # Aggregate by date
    df_agg = df.groupby('date')['sales'].sum().reset_index()
    last_date = df_agg['date'].max()
    
    # Create future data
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq='D'
    )
    
    last_row = df_agg.iloc[-1:].copy()
    future_data = pd.concat([last_row] * 30, ignore_index=True)
    future_data['date'] = future_dates
    
    # Add common business factors to future data if they don't exist
    # This allows us to test scenarios even if original data doesn't have these columns
    if 'promotion' not in future_data.columns:
        future_data['promotion'] = 0
    if 'price_discount_pct' not in future_data.columns:
        future_data['price_discount_pct'] = 0
    if 'marketing_spend' not in future_data.columns:
        future_data['marketing_spend'] = 0
    if 'competitor_promo' not in future_data.columns:
        future_data['competitor_promo'] = 0
    if 'stock_availability' not in future_data.columns:
        future_data['stock_availability'] = 100  # 100% available
    if 'weather_index' not in future_data.columns:
        future_data['weather_index'] = 0  # neutral weather
    
    # Define realistic business scenarios
    scenarios = {
        '1. Baseline (No Changes)': {},
        
        '2. Weekend Promotion (15% off)': {
            'promotion': 1,
            'price_discount_pct': 15
        },
        
        '3. Aggressive Marketing Campaign': {
            'marketing_spend': 5000,
            'promotion': 1
        },
        
        '4. Major Sale Event (30% off)': {
            'promotion': 1,
            'price_discount_pct': 30,
            'marketing_spend': 10000
        },
        
        '5. Competitor Running Promotion': {
            'competitor_promo': 1,
            'price_discount_pct': -10  # We lose sales, negative impact
        },
        
        '6. Stock Shortage (50% availability)': {
            'stock_availability': -50  # 50% reduction
        },
        
        '7. Payday Week Boost': {
            'marketing_spend': 2000,
            'promotion': 1,
            'price_discount_pct': 10
        },
        
        '8. Holiday Season Peak': {
            'promotion': 1,
            'marketing_spend': 8000,
            'price_discount_pct': 20,
            'weather_index': 5  # Good weather
        },
        
        '9. Economic Slowdown': {
            'price_discount_pct': -15,  # Negative sales impact
            'marketing_spend': -2000
        },
        
        '10. Perfect Storm (All Positive)': {
            'promotion': 1,
            'price_discount_pct': 25,
            'marketing_spend': 15000,
            'stock_availability': 20,  # Extra stock
            'weather_index': 10
        },
        
        '11. Worst Case (All Negative)': {
            'competitor_promo': 1,
            'stock_availability': -70,
            'weather_index': -10,
            'price_discount_pct': -20
        }
    }
    
    print("\n" + "="*60)
    print("RUNNING BUSINESS SCENARIO ANALYSIS")
    print("="*60)
    print(f"\nAnalyzing {len(scenarios)} different scenarios...")
    print("Note: Scenarios use hypothetical causal factors")
    print("Add actual data columns for more accurate predictions\n")
    
    # Run predictions for all scenarios
    scenario_results = {}
    
    for scenario_name, changes in scenarios.items():
        scenario_data = future_data.copy()
        
        # Apply changes
        for feature, change in changes.items():
            if feature in scenario_data.columns:
                scenario_data[feature] = scenario_data[feature] + change
        
        # Engineer features
        scenario_features = forecaster.engineer_features(scenario_data)
        
        # Ensure all required features exist
        for col in forecaster.feature_columns:
            if col not in scenario_features.columns:
                scenario_features[col] = 0
        
        # Select and scale features
        X_scenario = scenario_features[forecaster.feature_columns].fillna(scenario_features[forecaster.feature_columns].mean())
        X_scenario_scaled = forecaster.scaler.transform(X_scenario)
        
        # Predict
        predictions = forecaster.model.predict(X_scenario_scaled)
        scenario_results[scenario_name] = predictions
    
    # Display results
    print("\n" + "="*60)
    print("30-DAY FORECAST BY SCENARIO")
    print("="*60)
    print(f"{'Scenario':<40} | {'Total Sales':<15} | {'Avg Daily':<12} | {'vs Baseline':<12}")
    print("-" * 90)
    
    baseline_total = scenario_results['1. Baseline (No Changes)'].sum()
    
    for scenario_name, predictions in scenario_results.items():
        total = predictions.sum()
        avg = predictions.mean()
        
        if scenario_name == '1. Baseline (No Changes)':
            change_pct = 0
            print(f"{scenario_name:<40} | ${total:>13,.2f} | ${avg:>10,.2f} | {'--':>11}")
        else:
            change_pct = ((total - baseline_total) / baseline_total) * 100
            print(f"{scenario_name:<40} | ${total:>13,.2f} | ${avg:>10,.2f} | {change_pct:>+10.1f}%")
    
    # Find best and worst scenarios
    scenario_totals = {name: preds.sum() for name, preds in scenario_results.items() if name != '1. Baseline (No Changes)'}
    best_scenario = max(scenario_totals.items(), key=lambda x: x[1])
    worst_scenario = min(scenario_totals.items(), key=lambda x: x[1])
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print(f"📈 Best Scenario:  {best_scenario[0]}")
    print(f"   Potential Revenue: ${best_scenario[1]:,.2f}")
    print(f"   Uplift: +{((best_scenario[1] - baseline_total) / baseline_total * 100):.1f}%")
    print()
    print(f"📉 Worst Scenario: {worst_scenario[0]}")
    print(f"   Potential Revenue: ${worst_scenario[1]:,.2f}")
    print(f"   Impact: {((worst_scenario[1] - baseline_total) / baseline_total * 100):.1f}%")
    
    # Save detailed results
    detailed_results = pd.DataFrame({
        'date': future_dates
    })
    
    for scenario_name, predictions in scenario_results.items():
        col_name = scenario_name.replace('.', '').replace(' ', '_').replace('(', '').replace(')', '')
        detailed_results[col_name] = predictions
    
    detailed_results.to_csv('scenario_predictions_detailed.csv', index=False)
    print(f"\n✓ Detailed predictions saved to: scenario_predictions_detailed.csv")
    
    # Summary by scenario
    summary = pd.DataFrame({
        'Scenario': list(scenario_results.keys()),
        'Total_Sales': [preds.sum() for preds in scenario_results.values()],
        'Average_Daily': [preds.mean() for preds in scenario_results.values()],
        'Min_Daily': [preds.min() for preds in scenario_results.values()],
        'Max_Daily': [preds.max() for preds in scenario_results.values()]
    })
    
    summary.to_csv('scenario_summary.csv', index=False)
    print(f"✓ Scenario summary saved to: scenario_summary.csv")
    
    return scenario_results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("SALES FORECASTING - PREDICTION MODE")
    print("="*60)
    print("\nOptions:")
    print("1. Predict next 30 days")
    print("2. Predict next 60 days")
    print("3. Predict next 90 days")
    print("4. Custom prediction period")
    print("5. Scenario analysis")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            predict_future_sales(days_ahead=30)
        elif choice == '2':
            predict_future_sales(days_ahead=60)
        elif choice == '3':
            predict_future_sales(days_ahead=90)
        elif choice == '4':
            days = int(input("Enter number of days to predict: "))
            predict_future_sales(days_ahead=days)
        elif choice == '5':
            predict_with_custom_scenarios()
        else:
            print("Invalid choice. Running default 30-day prediction...")
            predict_future_sales(days_ahead=30)
            
    except KeyboardInterrupt:
        print("\n\nPrediction cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nRunning default 30-day prediction...")
        predict_future_sales(days_ahead=30)