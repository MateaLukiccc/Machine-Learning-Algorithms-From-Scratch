// NaiveBayes.cpp
#include "NaiveBayes.h"
#include <numeric>

using namespace std;

// constructor
NaiveBayes::NaiveBayes(double alpha) : alpha(alpha) {}

map<string, double> NaiveBayes::calculate_occurences(vector<string> class_column) {
    map<string, double> res;
    for (const auto& val : class_column) {
        res[val] += 1;
    }
    return res;
}

map<string, double> NaiveBayes::calculate_priors(map<string, double> raw_occurences, int data_size) {
    for (auto& [class_name, occurences] : raw_occurences) {
        occurences = log(occurences / data_size);
    }
    return raw_occurences;
}

double NaiveBayes::gaussian_prob(double x, double mean, double var) {
    double coeff = 1.0 / sqrt(2 * pi * var);
    double exponent = -((x - mean) * (x - mean)) / (2 * var);
    return coeff * exp(exponent);
}

void NaiveBayes::learn(string dataset_path, string target_column_name) {
    rapidcsv::Document data(dataset_path);
    vector<string> target_col = data.GetColumn<string>(target_column_name);
    int data_size = target_col.size();

    map<string, double> raw_occurences = calculate_occurences(target_col);
    class_counts = raw_occurences;
    priors = calculate_priors(raw_occurences, data_size);

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
                for (int i = 0; i < data_size; i++) {
                    if (target_col[i] == label) {
                        numeric_values.push_back(stod(feature_values[i]));
                    }
                }
                double mean = accumulate(numeric_values.begin(), numeric_values.end(), 0.0) / numeric_values.size();
                double variance = 0.0;
                for (double v : numeric_values) variance += (v - mean) * (v - mean);
                variance /= max(1.0, numeric_values.size() - 1.0);
                variance = max(variance, 1e-9);
                numerical_likelihoods[feature][label] = make_pair(mean, variance);
            }
            else {
                map<string, int> value_counts;
                for (int i = 0; i < data_size; i++) {
                    if (target_col[i] == label) {
                        value_counts[feature_values[i]]++;
                    }
                }
                set<string> unique_values(feature_values.begin(), feature_values.end());
                int V = unique_values.size();
                for (const auto& unique_value : unique_values) {
                    int count = value_counts[unique_value];
                    double probability = static_cast<double>(count + alpha) / (occurences + alpha * V);
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

void NaiveBayes::predict(string dataset_path, string target_column_name) {
    rapidcsv::Document data(dataset_path);
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
        cout << max_label << "\t" << maximum << "\n";
    }
}
