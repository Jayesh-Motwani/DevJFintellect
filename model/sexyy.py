# deepar_test_pipeline.py - Standalone Testing Pipeline

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Layer
import tensorflow as tf
import pandas as pd
import shap


class GaussianLayer(Layer):
    """
    Exact copy of your GaussianLayer - needed for model loading
    This must match your original implementation exactly
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the weights and biases."""
        n_weight_rows = input_shape[2]
        self.kernel_1 = self.add_weight(
            name="kernel_1",
            shape=(n_weight_rows, self.output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self.kernel_2 = self.add_weight(
            name="kernel_2",
            shape=(n_weight_rows, self.output_dim),
            initializer=glorot_normal(),
            trainable=True,
        )
        self.bias_1 = self.add_weight(
            name="bias_1",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.bias_2 = self.add_weight(
            name="bias_2",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
        )
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        """Do the layer computation."""
        output_mu = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.softplus(output_sig) + 1e-6
        return [output_mu, output_sig_pos]

    def compute_output_shape(self, input_shape):
        """Calculate the output dimensions."""
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]

    def get_config(self):
        """Required for proper serialization"""
        config = super().get_config()
        config.update({'output_dim': self.output_dim})
        return config


def gaussian_likelihood(y_true, y_pred):
    """
    Exact copy of your gaussian_likelihood function - needed for model loading
    """
    mu = y_pred[..., 0]
    sigma = y_pred[..., 1]

    # Safe clipping
    sigma = tf.clip_by_value(sigma, 1e-3, 1e2)

    # Reshape target to match mu shape
    y_true = tf.squeeze(y_true, axis=-1)
    y_true = tf.reshape(y_true, tf.shape(mu))

    nll = (
            0.5 * tf.math.log(2.0 * math.pi)
            + tf.math.log(sigma)
            + tf.square(y_true - mu) / (2.0 * tf.square(sigma))
    )

    tf.print("loss step range:", tf.reduce_min(nll), tf.reduce_max(nll))

    return tf.reduce_mean(nll)


class DeepARTestPipeline:
    def __init__(self, test_data_path, model_path, scaler_path, config_path):
        """
        Standalone DeepAR testing pipeline with custom layer support
        """

        self.test_data_path = test_data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path

        # Load configuration
        print("Loading model configuration...")
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        self.context_length = self.config['context_length']
        self.prediction_length = self.config['prediction_length']
        self.n_steps = self.config['n_steps']
        self.dimensions = self.config['dimensions']

        # Load trained scaler
        print("Loading trained scaler.")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load test data
        print("Loading test data.")
        with open(test_data_path, 'r') as f:
            self.test_data = json.load(f)

        print("Loading DeepAR model with custom objects like learnt scaler and configurations.")

        # Define custom objects dictionary
        custom_objects = {
            'GaussianLayer': GaussianLayer,
            'gaussian_likelihood': gaussian_likelihood,
            'main_output': GaussianLayer  # Alternative name that might be used
        }

        try:
            self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print(" Model loaded successfully!")
        except Exception as e:
            print(f" Error loading model: {e}")
            print("Trying alternative loading method...")
            # Try with custom object scope
            with tf.keras.utils.custom_object_scope(custom_objects):
                self.model = load_model(model_path, compile=False)
            print(" Model loaded with custom object scope!")

        print(f"\n Pipeline Summary:")
        print(f"  - Test samples: {len(self.test_data)}")
        print(f"  - Context length: {self.context_length}")
        print(f"  - Model dimensions: {self.dimensions}")
        print(f"  - Model file: {model_path}")
        print(f"  - Scaler loaded: ")
        print(f"  - Model loaded: ")

    def safe_log_returns(self, prices):
        """Calculate safe log returns - IDENTICAL to training preprocessing"""
        raw_price_series = np.array(prices)
        epsilon = 1e-8
        safe_prices = np.maximum(raw_price_series, epsilon)
        price_ratios = safe_prices[1:] / safe_prices[:-1]
        price_ratios = np.clip(price_ratios, epsilon, 1 / epsilon)
        log_returns = np.log(price_ratios)
        log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.1, neginf=-0.1)
        return log_returns

    def preprocess_sample(self, sample):
        """Preprocess single test sample - IDENTICAL to training preprocessing"""
        log_returns = self.safe_log_returns(sample['target'])
        dynamic_features = np.array(sample['feat_dynamic_real'])
        scaled_features = self.scaler.transform(dynamic_features.T).T
        price_feature = np.insert(log_returns, 0, 0)
        full_features = np.vstack([price_feature.reshape(1, -1), scaled_features])
        return log_returns, full_features

    def create_test_sequences(self):
        """Create test sequences for evaluation"""
        print("Creating test sequences...")
        sequences = []
        targets = []
        actual_prices = []

        for sample_idx, sample in enumerate(self.test_data):
            log_returns, full_features = self.preprocess_sample(sample)
            total_length = len(log_returns)

            if total_length <= self.context_length:
                print(f"  Warning: Sample {sample_idx} too short ({total_length} <= {self.context_length})")
                continue

            for start_idx in range(0, total_length - self.context_length):
                x_series = full_features[:, start_idx:start_idx + self.context_length].T
                y_target = log_returns[start_idx + self.context_length - 1:start_idx + self.context_length]

                if len(y_target) > 0:
                    sequences.append(x_series)
                    targets.append(y_target[0])
                    actual_prices.append({
                        'sample_idx': sample_idx,
                        'start_idx': start_idx,
                        'target_log_return': y_target[0]
                    })

        print(f" Created {len(sequences)} test sequences from {len(self.test_data)} samples")
        return np.array(sequences), np.array(targets), actual_prices

    def generate_predictions(self, sequences):
        """Generate predictions from the model"""
        print(f" Generating predictions for {len(sequences)} sequences...")

        if len(sequences) == 0:
            print(" ERROR: No sequences to predict!")
            return np.array([]), np.array([]), np.array([])

        print(f"Input shape: {sequences.shape}")

        # Get model predictions
        try:
            predictions = self.model.predict(sequences, batch_size=32, verbose=1)
            print(f" Predictions generated successfully!")
            print(f"Output shape: {predictions.shape}")
        except Exception as e:
            print(f" Error during prediction: {e}")
            print("Trying with smaller batch size. ")
            predictions = self.model.predict(sequences, batch_size=8, verbose=1)

        # Split into mu and sigma (assuming concatenated output)
        mu = predictions[:, :, 0]  # Mean predictions
        sigma = predictions[:, :, 1]  # Uncertainty predictions

        # Use the last time step for point forecasts
        mu_forecast = mu[:, -1]
        sigma_forecast = sigma[:, -1]

        # Ensure sigma is positive and reasonable
        sigma_forecast = np.maximum(sigma_forecast, 1e-6)
        sigma_forecast = np.minimum(sigma_forecast, 10.0)

        print(f" Prediction Summary:")
        print(f"  - μ range: [{np.min(mu_forecast):.4f}, {np.max(mu_forecast):.4f}]")
        print(f"  - σ range: [{np.min(sigma_forecast):.4f}, {np.max(sigma_forecast):.4f}]")

        # Sample from the learned distributions
        sampled_predictions = np.random.normal(
            loc=mu_forecast,
            scale=np.sqrt(np.maximum(sigma_forecast, 1e-8))
        )

        return mu_forecast, sigma_forecast, sampled_predictions

    def calculate_metrics(self, targets, predictions):
        """Calculate evaluation metrics"""
        if len(targets) == 0 or len(predictions) == 0:
            return {'error': 'No data for metrics calculation'}

        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)

        # Directional accuracy
        direction_correct = np.mean(np.sign(targets) == np.sign(predictions)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'Directional_Accuracy_%': direction_correct,
            'Num_Predictions': len(targets)
        }

    def run_evaluation(self):
        """Run complete evaluation pipeline"""

        sequences, targets, price_info = self.create_test_sequences()

        if len(sequences) == 0:
            print(" ERROR: No valid sequences created from test data!")
            print("Check that your test data has sufficient length per sample.")
            return None

        # Generate predictions
        mu_pred, sigma_pred, sampled_pred = self.generate_predictions(sequences)

        if len(mu_pred) == 0:
            print(" ERROR: No predictions generated!")
            return None

        print("\n Calculating evaluation metrics.")
        metrics = self.calculate_metrics(targets, mu_pred)

        print(" EVALUATION RESULTS")
        for metric, value in metrics.items():
            if isinstance(value, float) and metric != 'MAPE':
                print(f"{metric:25}: {value:.6f}")
            elif metric == 'MAPE':
                if value == float('inf'):
                    print(f"{metric:25}: N/A (division by zero)")
                else:
                    print(f"{metric:25}: {value:.2f}%")
            else:
                print(f"{metric:25}: {value}")

        # Create comprehensive results
        results = {
            'metrics': metrics,
            'predictions': {
                'mu': mu_pred.tolist(),
                'sigma': sigma_pred.tolist(),
                'sampled': sampled_pred.tolist(),
                'targets': targets.tolist()
            },
            'config': self.config,
            'test_info': {
                'num_test_samples': len(self.test_data),
                'num_sequences': len(sequences),
                'sequence_length': self.context_length,
                'model_dimensions': self.dimensions
            }
        }

        # Save detailed results
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(" Results saved to 'test_results.json'")

        # # Generate visualizations
        # print(" Creating visualizations...")
        # self.create_visualizations(targets, mu_pred, sigma_pred, sampled_pred)

        return results

    # def create_visualizations(self, targets, mu_pred, sigma_pred, sampled_pred):
    #     """Create comprehensive evaluation visualizations"""
    #     if len(targets) == 0:
    #         print("  No data to visualize")
    #         return
    #
    #     n_plot = min(200, len(targets))
    #
    #     plt.figure(figsize=(16, 12))
    #
    #     # Plot 1: Time Series Predictions
    #     plt.subplot(2, 3, 1)
    #     x_range = range(n_plot)
    #     plt.plot(x_range, targets[:n_plot], label='Actual', color='blue', alpha=0.8, linewidth=1.5)
    #     plt.plot(x_range, mu_pred[:n_plot], label='Predicted (μ)', color='red', alpha=0.8, linewidth=1.5)
    #
    #     # Confidence intervals
    #     upper_bound = (mu_pred + 1.96 * sigma_pred)[:n_plot]
    #     lower_bound = (mu_pred - 1.96 * sigma_pred)[:n_plot]
    #     plt.fill_between(x_range, lower_bound, upper_bound,
    #                      alpha=0.2, color='red', label='95% Confidence')
    #
    #     plt.title('DeepAR: Predictions vs Actual Values', fontsize=14, fontweight='bold')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Log Returns')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #
    #     # Plot 3: Residuals Analysis
    #     plt.subplot(2, 3, 3)
    #     residuals = targets[:n_plot] - mu_pred[:n_plot]
    #     plt.plot(residuals, alpha=0.7, color='purple', linewidth=1)
    #     plt.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    #     plt.axhline(y=np.mean(residuals), color='orange', linestyle=':', alpha=0.8, linewidth=2,
    #                 label=f'Mean: {np.mean(residuals):.4f}')
    #
    #     plt.title('Residuals Analysis', fontsize=14, fontweight='bold')
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Residual (Actual - Predicted)')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #     # Plot 4: Uncertainty Distribution
    #     plt.subplot(2, 3, 4)
    #     plt.hist(sigma_pred, bins=40, alpha=0.7, color='green', edgecolor='black', density=True)
    #     plt.axvline(np.mean(sigma_pred), color='red', linestyle='--', linewidth=2,
    #                 label=f'Mean σ: {np.mean(sigma_pred):.4f}')
    #     plt.axvline(np.median(sigma_pred), color='blue', linestyle=':', linewidth=2,
    #                 label=f'Median σ: {np.median(sigma_pred):.4f}')
    #
    #     plt.title('Predicted Uncertainty (σ) Distribution', fontsize=14, fontweight='bold')
    #     plt.xlabel('Sigma (Standard Deviation)')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #     # Plot 5: Prediction Intervals Coverage
    #     plt.subplot(2, 3, 5)
    #     # Calculate coverage of prediction intervals
    #     coverage_80 = np.mean((targets <= mu_pred + 1.28 * sigma_pred) &
    #                           (targets >= mu_pred - 1.28 * sigma_pred)) * 100
    #     coverage_95 = np.mean((targets <= mu_pred + 1.96 * sigma_pred) &
    #                           (targets >= mu_pred - 1.96 * sigma_pred)) * 100
    #
    #     coverages = [coverage_80, coverage_95]
    #     expected = [80, 95]
    #     intervals = ['80%', '95%']
    #
    #     x_pos = np.arange(len(intervals))
    #     plt.bar(x_pos, coverages, alpha=0.7, color=['orange', 'red'], label='Actual')
    #     plt.plot(x_pos, expected, 'ko-', linewidth=2, markersize=8, label='Expected')
    #
    #     plt.title('Prediction Interval Coverage', fontsize=14, fontweight='bold')
    #     plt.xlabel('Confidence Interval')
    #     plt.ylabel('Coverage (%)')
    #     plt.xticks(x_pos, intervals)
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #     # Add coverage percentages as text
    #     for i, (actual, exp) in enumerate(zip(coverages, expected)):
    #         plt.text(i, actual + 2, f'{actual:.1f}%', ha='center', fontweight='bold')
    #
    #     # Plot 6: Error Distribution
    #     plt.subplot(2, 3, 6)
    #     errors = targets - mu_pred
    #     plt.hist(errors, bins=40, alpha=0.7, color='red', edgecolor='black', density=True)
    #     plt.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero Error')
    #     plt.axvline(np.mean(errors), color='blue', linestyle=':', linewidth=2,
    #                 label=f'Mean Error: {np.mean(errors):.4f}')
    #
    #     plt.title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    #     plt.xlabel('Prediction Error')
    #     plt.ylabel('Density')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig('deepar_evaluation_plots.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    #     print(" Visualizations saved to 'deepar_evaluation_plots.png'")


def main():
    """Main execution function with enhanced error handling"""

    TEST_DATA_PATH = 'C:\\Users\\mjaye\\PycharmProjects\\DevJFintellect\\testing_data.json'  # Your test data file
    MODEL_PATH = 'C:\\Users\\mjaye\\PycharmProjects\\DevJFintellect\\model\\deepar_stock_forecaster.keras'
    SCALER_PATH = 'C:\\Users\\mjaye\\PycharmProjects\\DevJFintellect\\model\\trained_scaler.pkl'
    CONFIG_PATH = 'C:\\Users\\mjaye\\PycharmProjects\\DevJFintellect\\model\\model_config.pkl'

    required_files = {
        'Model': MODEL_PATH,
        'Scaler': SCALER_PATH,
        'Config': CONFIG_PATH,
        'Test Data': TEST_DATA_PATH
    }

    missing_files = []
    for name, path in required_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   {name}: {path} ({size:,} bytes)")
        else:
            print(f"   {name}: {path} (NOT FOUND)")
            missing_files.append(path)

    if missing_files:
        print(f"\n ERROR: Missing required files: {missing_files}")
        return

    # Run evaluation pipeline
    try:
        print("\n Initializing pipeline. ")
        pipeline = DeepARTestPipeline(
            test_data_path=TEST_DATA_PATH,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            config_path=CONFIG_PATH
        )

        # Run full evaluation
        print("\n Running evaluation. ")
        results = pipeline.run_evaluation()

        if results is not None:
            print(" SUCCESS! Testing pipeline completed successfully!")
            print("\n Generated files:")
            print(" test_results.json - Detailed metrics and predictions")

            # Quick summary
            metrics = results['metrics']
            print(f"\n Quick Summary:")
            print(f"  • RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
            print(f"  • MAE: {metrics.get('MAE', 'N/A'):.4f}")
            print(f"  • Directional Accuracy: {metrics.get('Directional_Accuracy_%', 'N/A'):.1f}%")
            print(f"  • Number of Predictions: {metrics.get('Num_Predictions', 'N/A')}")

        else:
            print("\n Testing pipeline failed! Check the error messages above.")

    except Exception as e:
        print(f"\n CRITICAL ERROR in testing pipeline:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\n Full traceback:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
