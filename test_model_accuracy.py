"""
Test Trading Model Accuracy
Backtests the model on historical data to verify effectiveness
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pytz

try:
    import joblib
    from tensorflow.keras.models import load_model
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca_config import AlpacaConfig
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)

class ModelTester:
    """Test model predictions against actual outcomes"""
    
    def __init__(self, model_path, scaler_path):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.lookback = 100
        
        # Load model and scaler
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        self.model = load_model(self.model_path, compile=False)
        self.scaler = joblib.load(self.scaler_path)
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Scaler loaded: {scaler_path}")
    
    def compute_features(self, df):
        """Compute features matching training.py"""
        if len(df) < self.lookback + 20:
            return None
        
        df = df.copy()
        
        # VWAP and std
        df['vwap'] = (df['close'] * df['volume']).rolling(window=self.lookback).sum() / \
                     df['volume'].rolling(window=self.lookback).sum()
        df['std'] = df['close'].rolling(window=self.lookback).std()
        
        # Z-score and dz
        df['z'] = (df['close'] - df['vwap']) / df['std']
        df['prev_z'] = df['z'].shift(1)
        df['dz'] = df['z'] - df['prev_z']
        df['avg_dz'] = df['dz'].rolling(window=5).mean()
        
        # MA slope
        df['ma_recent'] = df['close'].rolling(20).mean()
        df['ma_prev'] = df['close'].shift(10).rolling(20).mean()
        df['ma_slope'] = df['ma_recent'] - df['ma_prev']
        
        # Volume surge
        vol_mean = df['volume'].rolling(10).mean()
        df['vol_surge'] = (df['volume'] - vol_mean) / vol_mean
        
        # Deviation score
        df['dev_score'] = (df['vwap'] - df['close']) / df['std']
        
        # Trend
        df['trend'] = df['close'].rolling(window=20).apply(
            lambda x: np.polyfit(range(20), x, 1)[0] if len(x) == 20 else 0,
            raw=True
        )
        
        return df
    
    def predict(self, df, idx):
        """Get prediction for a specific row"""
        row = df.iloc[idx]
        features = np.array([
            row['z'], row['dz'], row['avg_dz'], row['ma_slope'],
            row['vol_surge'], row['dev_score'], row['trend']
        ], dtype=np.float32)
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = float(self.model.predict(scaled, verbose=0)[0, 0])
        
        return prediction
    
    def fetch_historical_data(self, symbol, days=5):
        """Fetch historical data from Alpaca"""
        print(f"\nFetching {days} days of data for {symbol}...")
        
        client = StockHistoricalDataClient(
            AlpacaConfig.API_KEY,
            AlpacaConfig.API_SECRET
        )
        
        now = datetime.now(pytz.UTC)
        start = now - timedelta(days=days)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=now,
            limit=10000
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if df.empty:
            raise ValueError("No data returned")
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level='symbol')
        
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap_raw']
        
        print(f"✓ Fetched {len(df)} bars")
        return df
    
    def backtest(self, df, buy_threshold=0.0002, hold_periods=5):
        """Backtest the model on historical data"""
        print(f"\nRunning backtest...")
        print(f"Buy threshold: {buy_threshold}")
        print(f"Hold periods: {hold_periods}")
        
        # Compute features
        df = self.compute_features(df)
        df = df.dropna()
        
        if len(df) < 100:
            raise ValueError("Insufficient data after feature computation")
        
        # Generate predictions
        predictions = []
        actuals = []
        signals = []
        
        for i in range(len(df) - hold_periods):
            pred = self.predict(df, i)
            
            # Actual return over hold period
            entry_price = df.iloc[i]['close']
            exit_price = df.iloc[i + hold_periods]['close']
            actual_return = (exit_price - entry_price) / entry_price
            
            predictions.append(pred)
            actuals.append(actual_return)
            signals.append(1 if pred > buy_threshold else 0)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        signals = np.array(signals)
        
        return predictions, actuals, signals, df
    
    def analyze_results(self, predictions, actuals, signals):
        """Analyze backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        # Overall statistics
        print(f"\nTotal predictions: {len(predictions)}")
        print(f"Buy signals: {signals.sum()} ({signals.mean()*100:.1f}%)")
        
        # Correlation
        correlation = np.corrcoef(predictions, actuals)[0, 1]
        print(f"\nPrediction-Actual Correlation: {correlation:.4f}")
        
        # Signal performance
        if signals.sum() > 0:
            signal_returns = actuals[signals == 1]
            no_signal_returns = actuals[signals == 0]
            
            print(f"\n--- When Model Says BUY ---")
            print(f"Trades: {len(signal_returns)}")
            print(f"Avg Return: {signal_returns.mean()*100:.4f}%")
            print(f"Win Rate: {(signal_returns > 0).mean()*100:.1f}%")
            print(f"Best: {signal_returns.max()*100:.2f}%")
            print(f"Worst: {signal_returns.min()*100:.2f}%")
            print(f"Std Dev: {signal_returns.std()*100:.4f}%")
            
            print(f"\n--- When Model Says NO BUY ---")
            print(f"Trades: {len(no_signal_returns)}")
            print(f"Avg Return: {no_signal_returns.mean()*100:.4f}%")
            
            # Edge calculation
            edge = signal_returns.mean() - no_signal_returns.mean()
            print(f"\n--- Model Edge ---")
            print(f"Edge: {edge*100:.4f}% per trade")
            
            # Sharpe-like ratio
            if signal_returns.std() > 0:
                sharpe = signal_returns.mean() / signal_returns.std()
                print(f"Return/Risk Ratio: {sharpe:.4f}")
            
            # Statistical significance
            from scipy import stats
            if len(signal_returns) > 30:
                t_stat, p_value = stats.ttest_1samp(signal_returns, 0)
                print(f"\nStatistical Test (t-test):")
                print(f"T-statistic: {t_stat:.4f}")
                print(f"P-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("✓ Results are statistically significant (p < 0.05)")
                else:
                    print("⚠ Results are NOT statistically significant")
        
        # Prediction distribution
        print(f"\n--- Prediction Distribution ---")
        print(f"Min: {predictions.min():.6f}")
        print(f"25th percentile: {np.percentile(predictions, 25):.6f}")
        print(f"Median: {np.median(predictions):.6f}")
        print(f"75th percentile: {np.percentile(predictions, 75):.6f}")
        print(f"Max: {predictions.max():.6f}")
        
        # Model effectiveness rating
        print("\n" + "="*60)
        print("MODEL EFFECTIVENESS RATING")
        print("="*60)
        
        score = 0
        max_score = 5
        
        if correlation > 0.1:
            score += 1
            print("✓ Positive correlation with actual returns")
        else:
            print("❌ Poor correlation with actual returns")
        
        if signals.sum() > 0 and signal_returns.mean() > 0:
            score += 1
            print("✓ Positive average return on buy signals")
        else:
            print("❌ Negative or zero average return")
        
        if signals.sum() > 0 and (signal_returns > 0).mean() > 0.5:
            score += 1
            print("✓ Win rate above 50%")
        else:
            print("❌ Win rate below 50%")
        
        if signals.sum() > 0 and edge > 0:
            score += 1
            print("✓ Model has positive edge")
        else:
            print("❌ No positive edge detected")
        
        if signals.sum() > 0 and len(signal_returns) > 30:
            t_stat, p_value = stats.ttest_1samp(signal_returns, 0)
            if p_value < 0.05 and signal_returns.mean() > 0:
                score += 1
                print("✓ Statistically significant positive returns")
            else:
                print("❌ Not statistically significant")
        
        print(f"\nOverall Score: {score}/{max_score}")
        
        if score >= 4:
            print("🎉 EXCELLENT - Model shows strong predictive power")
        elif score >= 3:
            print("✓ GOOD - Model shows promise but could be improved")
        elif score >= 2:
            print("⚠ FAIR - Model has some signal but needs work")
        else:
            print("❌ POOR - Model needs significant improvement")
        
        return score

def main():
    """Run model accuracy tests"""
    print("\n" + "="*60)
    print("TRADING MODEL ACCURACY TEST")
    print("="*60)
    
    # Configuration
    MODEL_PATH = "model/instinct_model.keras"
    SCALER_PATH = "model/scaler.pkl"
    SYMBOL = "AAPL"
    DAYS = 5
    
    try:
        # Initialize tester
        tester = ModelTester(MODEL_PATH, SCALER_PATH)
        
        # Fetch data
        df = tester.fetch_historical_data(SYMBOL, days=DAYS)
        
        # Run backtest
        predictions, actuals, signals, df_features = tester.backtest(df)
        
        # Analyze
        score = tester.analyze_results(predictions, actuals, signals)
        
        # Save results
        results_df = pd.DataFrame({
            'timestamp': df_features.iloc[:-5]['timestamp'].values,
            'close': df_features.iloc[:-5]['close'].values,
            'prediction': predictions,
            'actual_return': actuals,
            'signal': signals
        })
        
        output_file = f"backtest_results_{SYMBOL}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to: {output_file}")
        
        return 0 if score >= 3 else 1
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
