// masinsko_linearna.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "rapidcsv.h"
#include <vector>
#include <string>
#include <random>

using namespace std;

class LinearRegression {
private:
    vector<double> parameters;
    double lambda_reg = 0.1;
    vector<double> feature_means;
    vector<double> feature_stds;
    vector<string> feature_names;
    double initial_lr = 0.1;
    double decay_rate = 1e-4;
    int total_steps = 0;

    void standardize_column(rapidcsv::Document& data, const string& column_name) {
        vector<double> column_data = data.GetColumn<double>(column_name);
        double mean = 0.0;
        double var = 0.0;
        for (const double& v : column_data)
            mean += v;
        mean /= column_data.size();

        for (const double& v : column_data)
            var += (v - mean) * (v - mean);
        var /= column_data.size() - 1; // sample variance
        double std_dev = sqrt(var);
        std_dev = max(std_dev, 1e-9); // prevent zero standard deviation

        // Store standardization parameters
        feature_means.push_back(mean);
        feature_stds.push_back(std_dev);
        feature_names.push_back(column_name);

        // Standardize: (x - mean) / std_dev
        for (double& v : column_data) {
            v = (v - mean) / std_dev;
        }

        data.SetColumn(column_name, column_data);
    }

    void init_params(const int& n) {
        // set parameter values to random values
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(0.0, 1.0);
        parameters.resize(n);
        for (int i = 0; i < n; ++i) {
            parameters[i] = dis(gen);
        }
    }

    // Helper function to standardize a single feature vector
    vector<double> standardize_features(const vector<double>& raw_features) {
        vector<double> standardized_features;

        if (raw_features.size() != feature_means.size()) {
            cerr << "Error: Feature count mismatch! Expected " << feature_means.size()
                << " features, got " << raw_features.size() << endl;
            return raw_features; // Return original if size mismatch
        }

        for (size_t i = 0; i < raw_features.size(); ++i) {
            double standardized = (raw_features[i] - feature_means[i]) / feature_stds[i];
            standardized_features.push_back(standardized);
        }

        return standardized_features;
    }


public:
    LinearRegression(double lambda_reg = 0.0) : lambda_reg(lambda_reg) {}
    void learn(string dataset_path, string target_column_name) {
        rapidcsv::Document data(dataset_path);
        vector<double> targets = data.GetColumn<double>(target_column_name);
        vector<string> column_names = data.GetColumnNames();
        int instances = data.GetRowCount();
        int columns = data.GetColumnCount();
        for (string column_name : column_names) {
            if (column_name == target_column_name) {
                continue;
            }
            standardize_column(data, column_name);
        }
        vector<double> intercept_multiplier(instances, 1);
        data.InsertColumn(columns, intercept_multiplier, "X_0");
        data.RemoveColumn(target_column_name);

        init_params(columns);

        // we want X*wT since wT is nx1 the X need to be mxn and the result it mx1
        for (int iteration = 0; iteration < 10000; iteration++) {
            // Forward pass: compute predictions
            double learning_rate = initial_lr / (1 + decay_rate * total_steps);
            vector<double> prediction(instances, 0.0); // Use double for consistency
            for (int i = 0; i < instances; i++) {
                for (int j = 0; j < data.GetColumnCount(); j++) { // Use current column count
                    prediction[i] += data.GetCell<double>(j, i) * parameters[j];
                }
            }

            // Calculate MSE
            double mse = 0.0;
            for (int i = 0; i < instances; i++) {
                mse += (prediction[i] - targets[i]) * (prediction[i] - targets[i]);
            }
            mse /= instances; // Mean Squared Error

            // Calculate ridge penalty
            double penalty = 0.0;
            for (int i = 1; i < parameters.size(); i++) {
                penalty += parameters[i] * parameters[i];
            }
            mse += lambda_reg * penalty;

            // Gradient descent: compute gradients and update parameters
            vector<double> gradient(data.GetColumnCount(), 0.0);
            for (int j = 0; j < data.GetColumnCount(); j++) {
                for (int i = 0; i < instances; i++) {
                    double err = prediction[i] - targets[i];
                    gradient[j] += err * data.GetCell<double>(j, i); // Gradient for parameter j: (col, row)
                }
                gradient[j] /= instances; // Average gradient
                if (j != 0) {
                    gradient[j] += lambda_reg * parameters[j];
                }
                // Update parameter
                parameters[j] -= learning_rate * gradient[j];
            }

            // Calculate gradient norm (sum of absolute values)
            double grad_norm = 0.0;
            for (int j = 0; j < data.GetColumnCount(); j++) {
                grad_norm += abs(gradient[j]);
            }

            // Print progress and check for convergence
            cout << iteration << "\t" << grad_norm << "\t" << mse << endl;
            if (grad_norm < 0.01) {
                cout << "Converged at iteration " << iteration << endl;
                break;
            }
            total_steps++;
        }
    }
    void online_update(const vector<double>& raw_instance_features, double target_value) {
        // Standardize the input features using stored parameters
        vector<double> standardized_features = standardize_features(raw_instance_features);

        // instance_features size must match parameters size (including intercept)
        if (standardized_features.size() + 1 != parameters.size()) {
            cerr << "Feature size does not match parameters size!" << endl;
            return;
        }

        // Compute prediction for this instance
        double prediction = parameters[0]; // Start with intercept
        for (size_t j = 0; j < standardized_features.size(); ++j) {
            prediction += standardized_features[j] * parameters[j + 1];
        }

        double error = prediction - target_value;

        // Update parameters using SGD + L2 regularization (except w0)
        double learning_rate = initial_lr / (1 + decay_rate * total_steps);

        // Update intercept (no regularization)
        parameters[0] -= learning_rate * error;

        // Update other parameters (with regularization)
        for (size_t j = 0; j < standardized_features.size(); ++j) {
            double grad = error * standardized_features[j] + lambda_reg * parameters[j + 1];
            parameters[j + 1] -= learning_rate * grad;
        }
        total_steps++;
    }

    // Method to make predictions without updating parameters
    double predict(const vector<double>& raw_instance_features) {
        vector<double> standardized_features = standardize_features(raw_instance_features);

        if (standardized_features.size() + 1 != parameters.size()) {
            cerr << "Feature size does not match parameters size!" << endl;
            return 0.0;
        }

        double prediction = parameters[0]; // Start with intercept
        for (size_t j = 0; j < standardized_features.size(); ++j) {
            prediction += standardized_features[j] * parameters[j + 1];
        }

        return prediction;
    }
};


int main()
{
    LinearRegression lr = LinearRegression();
    lr.learn("boston.csv", "MEDV");
    vector<double> new_instance = { 0.02985,0,2.18,0,0.458,6.43,58.7,6.0622,3,222,18.7,394.12,5.21 };

    // First, make a prediction without updating
    double prediction = lr.predict(new_instance);
    cout << "Prediction for new instance: " << prediction << endl;

    // Then update with the actual target value
    lr.online_update(new_instance, 28.7);
    prediction = lr.predict(new_instance);
    cout << "Prediction for new instance: " << prediction << endl;

    return 0;

}

