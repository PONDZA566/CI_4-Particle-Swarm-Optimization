import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random seed
np.random.seed(42)

# Load dataset
data = pd.read_excel('AirQualityUCI.xlsx', decimal=',', na_values=-200)

# Drop rows with NaN values
data = data.dropna()

# Select attributes
X = data.iloc[:, [3, 6, 8, 10, 11, 12, 13]].values  # input attributes
y = data.iloc[:, 5].values  # output attribute (Benzene concentration)

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Function to predict benzene concentration (using a mock prediction for now)
def predict_benzene(X, best_hyperparams):
    return np.random.rand(len(X)) * 10  # Simulating predictions

# Function to calculate MAE
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Define a function for cross-validation
def cross_validate(X, y, n_folds=10):
    fold_size = len(X) // n_folds
    mae_per_fold = []
    all_predictions = np.zeros_like(y)

    for fold in range(n_folds):
        test_indices = range(fold * fold_size, (fold + 1) * fold_size)
        X_train = np.delete(X, test_indices, axis=0)
        y_train = np.delete(y, test_indices)
        X_test = X[test_indices]
        y_test = y[test_indices]

        # Train the MLP and get predictions
        hyperparams = best_hyperparams  # Assuming hyperparams are already optimized
        predictions = predict_benzene(X_test, hyperparams)

        all_predictions[test_indices] = predictions

        # Calculate MAE and store it
        mae = mean_absolute_error(y_test, predictions)
        mae_per_fold.append(mae)

    return mae_per_fold, all_predictions

# Mock function to simulate training process
def train_mlp(hyperparams, X_train, y_train):
    return 1.5  # Mock MAE value

# Define PSO for optimizing MLP
class Particle:
    def __init__(self, n_hidden_layers, n_nodes):
        self.position = np.random.rand(n_hidden_layers, n_nodes)
        self.velocity = np.random.rand(n_hidden_layers, n_nodes) * 0.1
        self.best_position = self.position.copy()
        self.best_error = float('inf')

def pso_optimization(n_particles, n_hidden_layers, n_nodes, iterations, X_train, y_train):
    particles = [Particle(n_hidden_layers, n_nodes) for _ in range(n_particles)]
    global_best_position = None
    global_best_error = float('inf')

    for _ in range(iterations):
        for particle in particles:
            error = train_mlp(particle.position, X_train, y_train)
            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = particle.position.copy()
            if error < global_best_error:
                global_best_error = error
                global_best_position = particle.position.copy()

            particle.velocity = particle.velocity + np.random.rand() * (particle.best_position - particle.position)
            particle.position += particle.velocity

    return global_best_position

# Hyperparameters for PSO
n_particles = 200
n_hidden_layers = 30  # Changeable
n_nodes = 10         # Changeable
iterations = 500

# Train an initial model to create training data for PSO optimization
X_train = X  
y_train = y

best_hyperparams = pso_optimization(n_particles, n_hidden_layers, n_nodes, iterations, X_train, y_train)

# Perform cross-validation to calculate MAE for each fold and predictions
mae_per_fold, all_predictions = cross_validate(X, y, n_folds=10)

# Print MAE for each fold
for fold, mae in enumerate(mae_per_fold, start=1):
    print(f'MAE for Fold {fold}: {mae}')

# Calculate and print overall MAE
overall_mae = np.mean(mae_per_fold)
print(f'Overall MAE across folds: {overall_mae}')

# Plotting MAE for each fold
folds = np.arange(1, len(mae_per_fold) + 1)

plt.figure(figsize=(12, 5))
plt.plot(folds, mae_per_fold, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('Mean Absolute Error (MAE) for Each Fold')
plt.xlabel('Fold Number')
plt.ylabel('MAE')
plt.xticks(folds)

plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting actual vs predicted values as a bar graph
plt.figure(figsize=(12, 6))

indices = np.arange(len(y))
bar_width = 0.35

plt.bar(indices, y, width=bar_width, label='Actual', color='b', alpha=0.6)
plt.bar(indices + bar_width, all_predictions, width=bar_width, label='Predicted', color='r', alpha=0.6)

plt.title('Actual vs Predicted Benzene Concentration')
plt.xlabel('Samples')
plt.ylabel('Benzene Concentration')
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
