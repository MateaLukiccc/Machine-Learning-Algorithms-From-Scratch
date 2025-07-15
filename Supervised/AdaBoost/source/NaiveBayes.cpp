// NaiveBayes.cpp
#include "NaiveBayes.h"
#include <numeric>

using namespace std;

// constructor
NaiveBayes::NaiveBayes(double alpha) : alpha(alpha) {}

// adjusted for weights
map<string, double> NaiveBayes::calculate_occurences(const vector<string>& class_column, const vector<double>& instance_weights) {
    map<string, double> res;
    for (size_t i = 0; i < class_column.size(); ++i) {
        res[class_column[i]] += instance_weights[i];;
    }
    return res;
}

// adjusted for weights
map<string, double> NaiveBayes::calculate_priors(map<string, double> raw_occurences, double total_weight) {
    for (auto& [class_name, weight_sum] : raw_occurences) {
        weight_sum = log(weight_sum / total_weight);
    }
    return raw_occurences;
}

double NaiveBayes::gaussian_prob(double x, double mean, double var) {
    double coeff = 1.0 / sqrt(2 * pi * var);
    double exponent = -((x - mean) * (x - mean)) / (2 * var);
    return coeff * exp(exponent);
}

void NaiveBayes::learn(string dataset_path, string target_column_name, vector<double> instance_weights) {
    rapidcsv::Document data(dataset_path);
    vector<string> target_col = data.GetColumn<string>(target_column_name);
    double total_weight = accumulate(instance_weights.begin(), instance_weights.end(), 0.0);
    int data_size = target_col.size();
    map<string, double> raw_occurences = calculate_occurences(target_col, instance_weights);
    class_counts = raw_occurences;
    priors = calculate_priors(raw_occurences, total_weight);

    vector<string> column_names = data.GetColumnNames();
    for (string feature : column_names) {
        if (feature == target_column_name) continue;
        vector<string> feature_values = data.GetColumn<string>(feature);
        bool is_numerical = true;
        for (const string& val : feature_values) {
            try { stod(val); }
            catch (...) { is_numerical = false; break; }
        }

        if (!is_numerical) {
            set<string> unique_values(feature_values.begin(), feature_values.end());
            feature_vocabulary_sizes[feature] = unique_values.size();
        }

        for (const auto& class_count_pair : raw_occurences) {
            string label = class_count_pair.first;
            double occurences = class_count_pair.second;

            if (is_numerical) {
                vector<double> numeric_values;
                vector<double> weights;
                for (int i = 0; i < data_size; i++) {
                    if (target_col[i] == label) {
                        numeric_values.push_back(stod(feature_values[i]));
                        weights.push_back(instance_weights[i]);
                    }
                }
                double weight_sum = accumulate(weights.begin(), weights.end(), 0.0);
                double mean = 0.0;
                for (size_t i = 0; i < numeric_values.size(); ++i)
                    mean += numeric_values[i] * weights[i];
                mean /= weight_sum;

                double variance = 0.0;
                for (size_t i = 0; i < numeric_values.size(); ++i)
                    variance += weights[i] * (numeric_values[i] - mean) * (numeric_values[i] - mean);
                variance /= max(1e-9, weight_sum); // Avoid division by zero
                variance = max(variance, 1e-9);
                numerical_likelihoods[feature][label] = make_pair(mean, variance);
            }
            else {
                map<string, double> value_counts;
                for (int i = 0; i < data_size; i++) {
                    if (target_col[i] == label) {
                        value_counts[feature_values[i]] += instance_weights[i];;
                    }
                }
                set<string> unique_values(feature_values.begin(), feature_values.end());
                int V = unique_values.size();
                for (const auto& unique_value : unique_values) {
                    double count = value_counts[unique_value];
                    double probability = (count + alpha) / (occurences + alpha * V);
                    likelihoods[feature][label][unique_value] = log(probability);
                }
            }
        }
    }
}

void NaiveBayes::print_priors() {
    cout << "Priors:\n";
    for (const auto& [class_name, prob] : priors) {
        cout << "  P(" << class_name << ") = " << prob << "\n";
    }
}

void NaiveBayes::print_likelihoods() {
    cout << "\nCategorical Likelihoods:\n";
    for (const auto& [feature_name, class_map] : likelihoods) {
        cout << "Feature: " << feature_name << "\n";
        for (const auto& [class_label, value_map] : class_map) {
            cout << "  Class: " << class_label << "\n";
            for (const auto& [feature_value, prob] : value_map) {
                cout << "    P(" << feature_name << "=" << feature_value << " | " << class_label << ") = " << prob << "\n";
            }
        }
    }

    cout << "\nNumerical Likelihoods:\n";
    for (const auto& [feature_name, class_map] : numerical_likelihoods) {
        cout << "Feature: " << feature_name << "\n";
        for (const auto& [class_label, mean_var] : class_map) {
            cout << "  Class: " << class_label
                << " -> mean = " << mean_var.first
                << ", variance = " << mean_var.second << "\n";
        }
    }
}

vector<int> NaiveBayes::predict(string dataset_path, string target_column_name) {
    rapidcsv::Document data(dataset_path);
    vector<int> predictions(data.GetRowCount());
    for (int row_index = 0; row_index < data.GetRowCount(); row_index++) {
        map<string, double> row_probability;
        for (const auto& prior : priors) {
            string label = prior.first;
            double probability = prior.second;

            for (int column_index = 0; column_index < data.GetColumnCount(); column_index++) {
                string feature = data.GetColumnName(column_index);
                if (feature == target_column_name) continue;
                string feature_value = data.GetColumn<string>(feature)[row_index];

                if (numerical_likelihoods.find(feature) != numerical_likelihoods.end()) {
                    double x = stod(feature_value);
                    auto mean_var_it = numerical_likelihoods[feature].find(label);
                    if (mean_var_it == numerical_likelihoods[feature].end()) {
                        probability += log(1e-10);
                        continue;
                    }
                    double mean = mean_var_it->second.first;
                    double var = mean_var_it->second.second;
                    probability += log(gaussian_prob(x, mean, var));
                }
                else {
                    if (likelihoods.count(feature) &&
                        likelihoods[feature].count(label) &&
                        likelihoods[feature][label].count(feature_value)) {
                        probability += likelihoods[feature][label][feature_value];
                    }
                    else {
                        double class_count = class_counts[label];
                        int vocab_size = feature_vocabulary_sizes[feature];
                        double smoothed_prob = alpha / (class_count + alpha * (vocab_size + 1));
                        probability += log(smoothed_prob);
                    }
                }
            }
            row_probability[label] = probability;
        }

        double maximum = -INFINITY;
        string max_label;
        for (const auto& [label, prob] : row_probability) {
            if (prob > maximum) {
                maximum = prob;
                max_label = label;
            }
        }
        // cout << max_label << "\t" << maximum << "\n";
        
        predictions[row_index] = stoi(max_label);
    }
    return predictions;
}
