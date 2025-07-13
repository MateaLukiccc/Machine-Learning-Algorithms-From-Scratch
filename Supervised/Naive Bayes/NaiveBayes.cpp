// masinsko_bayes.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "rapidcsv.h"
#include <map>
#include <string>
#include <set>
#include <unordered_set>
#include <cmath>

using namespace std;

const double pi = 3.1415926535897932;

class NaiveBayes {
    private:
        map<string, double> priors; // p(c)
        map<string, map<string, map<string, double>>> likelihoods; // p(x|c)
        map<string, map<string, pair<double, double>>> numerical_likelihoods; // p(x|c)
        
        // smoothing fields
        double alpha; 
        map<string, double> class_counts;
        map<string, int> feature_vocabulary_sizes;

        map<string, double> calculate_occurences(vector<string> class_column) {
            /** 
            * @brief Calculates the occurences of each class in our string vector
            * <A, A, B, A, C> -> <"A": 3, "B": 1, "C": 1>
            */
            map<string, double> res;
            for (int i = 0; i < class_column.size(); i++) {
                res[class_column[i]] += 1;
            }

            return res;
        }

        map<string, double> calculate_priors(map<string, double> raw_occurences, int data_size) {
            /**
            * @brief makes probabilities from number of occurences
            * "A": 3, "B": 1 -> "A": 0.75, "B": 0.25 and apply log transformation
            */
            for (auto& [class_name, occurences] : raw_occurences) {
                occurences = log(occurences / data_size);
            }
            return raw_occurences;
        }

        double gaussian_prob(double x, double mean, double var) {
            double coeff = 1.0 / sqrt(2 * pi * var);
            double exponent = -((x - mean) * (x - mean)) / (2 * var);
            return coeff * exp(exponent);
        }
    public:
        NaiveBayes(double alpha = 1.0) : alpha(alpha) {}
        void learn(string dataset_path, string target_column_name) {
            /**
            * @brief Algoritham for calculating priors and likelihoods for Naive Bayes using Laplace Smoothing
            */
            rapidcsv::Document data(dataset_path);
            vector<string> target_col = data.GetColumn<string>(target_column_name);
            int data_size = target_col.size();
            map<string, double> raw_occurences = calculate_occurences(target_col);
            class_counts = raw_occurences;
            priors = calculate_priors(raw_occurences, data_size); // p(k)

            // p(x|k) = p(x1|k) * p(x2|k)*...*p(xp|k)
            vector<string> column_names = data.GetColumnNames();
            for (string feature : column_names) {
                if (feature == target_column_name) continue;
                vector<string> feature_values = data.GetColumn<string>(feature);

                // check type of the column
                bool is_numerical = true;
                for (const string& val : feature_values) {
                    try { stod(val); }
                    catch (...) { is_numerical = false; break; }
                }

                if (!is_numerical) {
                    // Store vocabulary size for this feature
                    set<string> unique_values(feature_values.begin(), feature_values.end());
                    feature_vocabulary_sizes[feature] = unique_values.size();
                }


                // for each class calculate conditional probabilities npr. p(x|class_1)
                for (const auto& class_count_pair : raw_occurences) {
                    string label = class_count_pair.first; 
                    double occurences = class_count_pair.second; 

                    if (is_numerical) {
                        // Compute mean and variance
                        vector<double> numeric_values;
                        for (int i = 0; i < data_size; i++) {
                            if (target_col[i] == label) {
                                numeric_values.push_back(stod(feature_values[i]));
                            }
                        }
                        double mean = 0.0;
                        for (double v : numeric_values) mean += v;
                        mean /= numeric_values.size();

                        double variance = 0.0;
                        for (double v : numeric_values) variance += (v - mean) * (v - mean);
                        variance /= numeric_values.size() - 1; // sample variance
                        variance = max(variance, 1e-9); // prevent zero variance
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
                        for (const auto& unique_value: unique_values) {
                            int count = value_counts[unique_value];
                            double probability = static_cast<double>(count + alpha) / (occurences + alpha*V);
                            likelihoods[feature][label][unique_value] = log(probability);
                        }
                    }
                }
            }
        }

        void print_priors() {
            cout << "Priors:" << "\n";
            for (const auto& [class_name, prob] : priors) {
                cout << "  P(" << class_name << ") = " << prob << "\n";
            }
        }

        void print_likelihoods() {
            cout << "\nCategorical Likelihoods:\n";
            for (const auto& [feature_name, class_map] : likelihoods) {
                cout << "Feature: " << feature_name << "\n";
                for (const auto& [class_label, value_map] : class_map) {
                    cout << "  Class: " << class_label << "\n";
                    for (const auto& [feature_value, prob] : value_map) {
                        cout << "    P(" << feature_name << "=" << feature_value
                            << " | " << class_label << ") = " << prob << "\n";
                    }
                }
                cout << "\n";
            }


            cout << "\nNumerical Likelihoods (Gaussian parameters):\n";
            for (const auto& [feature_name, class_map] : numerical_likelihoods) {
                cout << "Feature: " << feature_name << "\n";
                for (const auto& [class_label, mean_var] : class_map) {
                    double mean = mean_var.first;
                    double variance = mean_var.second;
                    cout << "  Class: " << class_label << "\n";
                    cout << "    mean = " << mean << ", variance = " << variance << "\n";
                }
                cout << "\n";
            }
        }

        void predict(string dataset_path, string target_column_name) {
            rapidcsv::Document data(dataset_path);
            map<string, double> result;
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
                            // unseen numerical value
                            if (mean_var_it == numerical_likelihoods[feature].end()) {
                                probability += log(1e-10); // small value
                                continue;
                            }

                            double mean = mean_var_it->second.first;
                            double var = mean_var_it->second.second;

                            double gauss_prob = log(gaussian_prob(x, mean, var));
                            probability += gauss_prob;
                        }
                        else {
                            if (likelihoods.count(feature) &&
                                likelihoods[feature].count(label) &&
                                likelihoods[feature][label].count(feature_value)) {

                                probability += likelihoods[feature][label][feature_value];
                            }
                            else {
                                // Unseen value 
                                double class_count = class_counts[label];
                                int vocab_size = feature_vocabulary_sizes[feature];

                                // For unseen values, we need to account for the expanded vocabulary
                                // P(unseen_value|class) = alpha / (class_count + alpha * (vocab_size + 1))
                                double smoothed_prob = alpha / (class_count + alpha * (vocab_size + 1));
                                probability += log(smoothed_prob);
                            }
                        }
                    }
                    row_probability[label] = probability;
                }
                double maximum = -INFINITY;
                string max_label;
                // check for biggest probability
                for (const auto& probability : row_probability) {
                    if (maximum == -1 || maximum < probability.second) {
                        max_label = probability.first;
                        maximum = probability.second;
                    }
                }
                cout << max_label << "\t" << maximum << endl;
            }
        }
};

int main()
{
    NaiveBayes nb = NaiveBayes();
    nb.learn("prehlada.csv", "Prehlada");
    nb.print_priors();
    nb.print_likelihoods();
    nb.predict("prehlada_novi.csv", "Prehlada");
}
