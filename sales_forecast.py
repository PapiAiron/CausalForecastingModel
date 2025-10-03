"""
Causal Sales Forecasting Model
This model predicts sales based on causal factors like economics, weather, promotions, etc.
Configured for: dummy_sales_data_2_years.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CausalSalesForecast:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        
    def load_data(self, filepath):
        """
        Load sales data from CSV file
        Expected columns: date, sales, and causal factors
        """
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Try to identify date column (common names)
        date_columns = ['date', 'Date', 'DATE', 'datetime', 'timestamp', 'time']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            if date_col != 'date':
                df = df.drop(columns=[date_col])
        else:
            print("Warning: No date column found. Using index as date.")
            df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='D')
        
        # Remove rows with missing dates
        before_drop = len(df)
        df = df.dropna(subset=['date'])
        after_drop = len(df)
        if before_drop != after_drop:
            print(f"Removed {before_drop - after_drop} rows with missing dates")
        
        # Try to identify sales/value column
        value_columns = ['Value', 'value', 'sales', 'Sales', 'revenue', 'Revenue', 'amount', 'Amount']
        value_col = None
        for col in value_columns:
            if col in df.columns:
                value_col = col
                break
        
        # Convert value column to numeric
        if value_col:
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            # Rename to 'sales' for consistency
            if value_col != 'sales':
                df['sales'] = df[value_col]
                df = df.drop(columns=[value_col])
        
        # Remove rows with missing or zero sales
        df = df.dropna(subset=['sales'])
        df = df[df['sales'] > 0]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head(10))
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Show unique outlets and SKUs if they exist
        if 'OUTLET' in df.columns:
            print(f"\nUnique Outlets: {df['OUTLET'].nunique()}")
            print(f"Outlets: {df['OUTLET'].unique()[:10]}")
        if 'SKU' in df.columns:
            print(f"\nUnique SKUs: {df['SKU'].nunique()}")
            print(f"Top 10 SKUs: {df['SKU'].value_counts().head(10)}")
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional time-based and interaction features
        """
        df = df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Find sales column (common names)
        sales_col = None
        for col in ['sales', 'Sales', 'SALES', 'revenue', 'Revenue', 'amount', 'total_sales']:
            if col in df.columns:
                sales_col = col
                break
        
        # Lag features for sales (if available)
        if sales_col and sales_col in df.columns:
            df['sales_lag_1'] = df[sales_col].shift(1)
            df['sales_lag_7'] = df[sales_col].shift(7)
            df['sales_lag_30'] = df[sales_col].shift(30)
            df['sales_rolling_7'] = df[sales_col].rolling(window=7, min_periods=1).mean()
            df['sales_rolling_30'] = df[sales_col].rolling(window=30, min_periods=1).mean()
            df['sales_rolling_std_7'] = df[sales_col].rolling(window=7, min_periods=1).std()
        
        # Interaction features (check if columns exist)
        if 'temperature' in df.columns and 'is_weekend' in df.columns:
            df['temp_weekend_interaction'] = df['temperature'] * df['is_weekend']
        
        if 'promotion' in df.columns and 'is_weekend' in df.columns:
            df['promo_weekend_interaction'] = df['promotion'] * df['is_weekend']
        
        if 'economic_index' in df.columns:
            df['economic_seasonal'] = df['economic_index'] * df['month_sin']
        
        # Weather interactions
        if 'temperature' in df.columns and 'rainfall' in df.columns:
            df['weather_comfort'] = df['temperature'] * (1 - df['rainfall'] / df['rainfall'].max())
        
        return df
    
    def prepare_data(self, df, target_col='sales', test_size=0.2, aggregate_by_date=True):
        """
        Prepare data for training
        """
        print(f"\nUsing '{target_col}' as target variable")
        
        # If we have multiple rows per date (outlets/SKUs), aggregate them
        if aggregate_by_date and 'date' in df.columns:
            rows_before = len(df)
            df_agg = df.groupby('date').agg({
                target_col: 'sum',  # Sum sales across all outlets/SKUs per day
            }).reset_index()
            
            # Add any other numeric columns by averaging
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != target_col and col in df.columns:
                    df_agg[col] = df.groupby('date')[col].mean().values
            
            print(f"Aggregated {rows_before} rows to {len(df_agg)} daily records")
            df = df_agg
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Remove rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Define feature columns (exclude date and target)
        exclude_cols = ['date', target_col]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\nFeatures being used: {len(self.feature_columns)}")
        print(f"Feature names: {self.feature_columns[:10]}..." if len(self.feature_columns) > 10 else f"Feature names: {self.feature_columns}")
        
        # Handle missing values in features
        df[self.feature_columns] = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        
        X = df[self.feature_columns]
        y = df[target_col]
        
        # Time series split to maintain temporal order
        split_idx = int(len(df) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nTrain period: {df['date'].iloc[0]} to {df['date'].iloc[split_idx-1]}")
        print(f"Test period: {df['date'].iloc[split_idx]} to {df['date'].iloc[-1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_model(self, X_train, y_train, model_type='xgboost'):
        """
        Train the forecasting model
        """
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("Model training completed!")
        return self.model
    
    def save_model(self, filepath='sales_forecast_model.pkl'):
        """
        Save the trained model, scaler, and feature columns to disk
        """
        if self.model is None:
            print("No model to save. Train a model first.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved to: {filepath}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")
        return filepath
    
    def load_model(self, filepath='sales_forecast_model.pkl'):
        """
        Load a trained model from disk
        """
        if not os.path.exists(filepath):
            print(f"Error: Model file '{filepath}' not found!")
            return False
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', None)
        
        print(f"\n✓ Model loaded from: {filepath}")
        print(f"  Features: {len(self.feature_columns)}")
        return True
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        """
        predictions = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        r2 = r2_score(y_test, predictions)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"MAE (Mean Absolute Error):     {mae:.2f}")
        print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
        print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")
        print(f"R² Score:                       {r2:.4f}")
        print("="*50)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'predictions': predictions
        }
    
    def predict_with_scenarios(self, base_data, scenarios):
        """
        Predict sales under different causal scenarios
        
        scenarios: dict with keys as scenario names and values as dict of feature changes
        Example: {'High Promotion': {'promotion': 1}, 'Cold Weather': {'temperature': -10}}
        """
        results = {}
        
        # Base prediction
        base_features = self.engineer_features(base_data.copy())
        base_features = base_features[self.feature_columns].fillna(base_features[self.feature_columns].mean())
        base_scaled = self.scaler.transform(base_features)
        base_pred = self.model.predict(base_scaled)
        results['base'] = base_pred
        
        # Scenario predictions
        for scenario_name, changes in scenarios.items():
            scenario_data = base_data.copy()
            
            # Apply changes
            for feature, change in changes.items():
                if feature in scenario_data.columns:
                    scenario_data[feature] = scenario_data[feature] + change
                else:
                    print(f"Warning: Feature '{feature}' not found in data. Skipping.")
            
            scenario_features = self.engineer_features(scenario_data)
            scenario_features = scenario_features[self.feature_columns].fillna(scenario_features[self.feature_columns].mean())
            scenario_scaled = self.scaler.transform(scenario_features)
            scenario_pred = self.model.predict(scenario_scaled)
            results[scenario_name] = scenario_pred
        
        return results
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance
        """
        if self.feature_importance is None:
            print("No feature importance available")
            return
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Top Feature Importances for Sales Prediction', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nSaved: feature_importance.png")
        plt.show()
    
    def plot_predictions(self, y_test, predictions, dates=None):
        """
        Plot actual vs predicted sales
        """
        plt.figure(figsize=(16, 6))
        
        if dates is not None:
            plt.plot(dates, y_test, label='Actual Sales', linewidth=2, marker='o', markersize=3)
            plt.plot(dates, predictions, label='Predicted Sales', linewidth=2, alpha=0.7, marker='s', markersize=3)
            plt.xticks(rotation=45)
        else:
            plt.plot(y_test.values, label='Actual Sales', linewidth=2)
            plt.plot(predictions, label='Predicted Sales', linewidth=2, alpha=0.7)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sales', fontsize=12)
        plt.title('Actual vs Predicted Sales - Test Set Performance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
        print("Saved: predictions.png")
        plt.show()
    
    def plot_scenario_comparison(self, scenario_results, dates=None):
        """
        Plot comparison of different scenarios
        """
        plt.figure(figsize=(14, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_results)))
        
        for idx, (scenario_name, predictions) in enumerate(scenario_results.items()):
            if dates is not None:
                plt.plot(dates, predictions, label=scenario_name, linewidth=2.5, alpha=0.8, color=colors[idx])
            else:
                plt.plot(predictions, label=scenario_name, linewidth=2.5, alpha=0.8, color=colors[idx])
        
        if dates is not None:
            plt.xticks(rotation=45)
        
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Predicted Sales', fontsize=12)
        plt.title('Sales Forecast - Scenario Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: scenario_comparison.png")
        plt.show()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    print("="*60)
    print("CAUSAL SALES FORECASTING MODEL")
    print("="*60)
    
    # Initialize the model
    forecaster = CausalSalesForecast()
    
    # Load your actual data
    try:
        df = forecaster.load_data('dummy_sales_data_2_years.csv')
    except FileNotFoundError:
        print("\nError: 'dummy_sales_data_2_years.csv' not found!")
        print("Please make sure the file is in the same directory as this script.")
        print("\nExpected file format:")
        print("date,sales,temperature,promotion,economic_index,...")
        exit()
    
    # Prepare data (auto-detects sales column)
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = forecaster.prepare_data(
        df, 
        target_col='sales',  # Now using 'sales' after conversion
        test_size=0.2,
        aggregate_by_date=True  # Aggregate multiple outlets/SKUs per day
    )
    
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Train model (you can change to 'random_forest' or 'gradient_boosting')
    forecaster.train_model(X_train, y_train, model_type='xgboost')
    
    # Save the trained model
    forecaster.save_model('sales_forecast_model.pkl')
    
    # Evaluate
    metrics = forecaster.evaluate_model(X_test, y_test)
    
    # Plot feature importance
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    print(forecaster.feature_importance.head(15).to_string(index=False))
    forecaster.plot_feature_importance(top_n=15)
    
    # Plot predictions
    test_dates = df.groupby('date')['sales'].sum().reset_index()['date'].iloc[-len(y_test):]
    forecaster.plot_predictions(y_test, metrics['predictions'], test_dates)
    
    # Scenario analysis
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS")
    print("="*60)
    
    # Create future dates for prediction
    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=30,
        freq='D'
    )
    
    # Get the last known aggregated values as base scenario
    df_agg = df.groupby('date')['sales'].sum().reset_index()
    last_row = df_agg.iloc[-1:].copy()
    future_data = pd.concat([last_row] * 30, ignore_index=True)
    future_data['date'] = future_dates
    
    # Define your scenarios based on available columns
    available_cols = df.columns.tolist()
    
    scenarios = {}
    
    # Add scenarios based on what columns exist in your data
    if 'promotion' in available_cols:
        scenarios['Active Promotion'] = {'promotion': 1}
    
    if 'temperature' in available_cols:
        scenarios['Hot Weather (+10°)'] = {'temperature': 10}
        scenarios['Cold Weather (-10°)'] = {'temperature': -10}
    
    if 'economic_index' in available_cols:
        scenarios['Economic Boom (+10%)'] = {'economic_index': 10}
        scenarios['Economic Downturn (-10%)'] = {'economic_index': -10}
    
    if 'competitor_price' in available_cols:
        scenarios['Competitive Pricing (-$5)'] = {'competitor_price': -5}
    
    # Combined scenario
    if len(scenarios) > 0:
        all_positive = {}
        if 'promotion' in available_cols:
            all_positive['promotion'] = 1
        if 'temperature' in available_cols:
            all_positive['temperature'] = 5
        if 'economic_index' in available_cols:
            all_positive['economic_index'] = 5
        
        if all_positive:
            scenarios['Best Case (All Positive)'] = all_positive
    
    if len(scenarios) == 0:
        print("No recognizable causal factors found for scenario analysis.")
        print("Available columns:", available_cols)
    else:
        # Predict scenarios
        scenario_results = forecaster.predict_with_scenarios(future_data, scenarios)
        
        # Display scenario results
        print("\nFuture 30-Day Sales Forecast by Scenario:")
        print("-" * 60)
        for scenario_name, predictions in scenario_results.items():
            avg_sales = np.mean(predictions)
            total_sales = np.sum(predictions)
            if scenario_name == 'base':
                print(f"{'Baseline (Current Conditions)':<35} | Avg: ${avg_sales:>10,.2f} | Total: ${total_sales:>12,.2f}")
            else:
                base_avg = np.mean(scenario_results['base'])
                change_pct = ((avg_sales - base_avg) / base_avg) * 100
                print(f"{scenario_name:<35} | Avg: ${avg_sales:>10,.2f} | Total: ${total_sales:>12,.2f} | Change: {change_pct:>+6.1f}%")
        
        # Plot scenario comparison
        forecaster.plot_scenario_comparison(scenario_results, future_dates)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ sales_forecast_model.pkl - Trained model (use for predictions)")
    print("  ✓ feature_importance.png - Shows which factors matter most")
    print("  ✓ predictions.png - Actual vs predicted sales comparison")
    if len(scenarios) > 0:
        print("  ✓ scenario_comparison.png - Future scenarios comparison")
    
    print("\nYou can now use this model to:")
    print("  • Predict future sales with: python use_saved_model.py")
    print("  • Understand what drives your sales")
    print("  • Test different business scenarios")
    print("  • Make data-driven decisions")
    print("="*60)
    