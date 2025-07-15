// masinsko_adaboost.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <iostream>
#include "rapidcsv.h"
#include "NaiveBayes.h"
#include <string>
#include <vector>
#include "WeakClassifier.h"
#include <memory>
#include <cmath>
#include <numeric>
#include <iomanip>

using namespace std;

class AdaBoost {
public:
    void learn(string dataset_path, string target_column, WeakClassifier& algorithm, double adaptation_rate = 1.0) {
        rapidcsv::Document data(dataset_path);
        int n_rows = data.GetRowCount();
        int m_columns = data.GetColumnCount();

        // instance weights
        vector<double> D_t(n_rows, 1.0 / n_rows);

        // reserve 5 places for weak classifiers
        int ensamble_size = 5;
        vector<unique_ptr<WeakClassifier>> ensemble;
        ensemble.reserve(5);

        // initialize model weights
        vector<double> alphas;

        // get -1 and 1 labels
        vector<int> labels = data.GetColumn<int>(target_column);

        // fit each classifier
        for (int classifier = 0; classifier < ensamble_size; classifier++) {
            std::unique_ptr<WeakClassifier> weak_classifier = algorithm.clone();
            weak_classifier->learn(dataset_path, target_column, D_t);
            vector<int> predictions = weak_classifier->predict(dataset_path, target_column);

            double weighted_error = 0.0;
            for (size_t i = 0; i < n_rows; ++i) {
                if (predictions[i] != labels[i]) {
                    weighted_error += D_t[i];
                }
            }

            // compute model weight (alpha)
            double alpha = 0.5 * log((1 - weighted_error) / weighted_error);
            alpha *= adaptation_rate;

            // store model and alpha
            ensemble.push_back(move(weak_classifier));
            alphas.push_back(alpha);

            // update instance weights
            for (size_t i = 0; i < n_rows; ++i) {
                D_t[i] *= exp(-alpha * predictions[i] * labels[i]);
            }

            // normalize weights
            double sum_D = accumulate(D_t.begin(), D_t.end(), 0.0);
            for (double& d : D_t) d /= sum_D;
        }
        // store results
        this->ensemble = move(ensemble);
        this->model_weights = move(alphas);

        cout << "Model weights (alphas):" << endl;
        for (auto x : model_weights) {
            cout << fixed << setprecision(4) << x << endl;
        }
    }

    vector<int> predict(string dataset_path, string target_column) {
        auto [predictions, confidence] = predict_with_confidence(dataset_path, target_column);
        return predictions;
    }

    // returns both predictions and confidence scores
    pair<vector<int>, vector<double>> predict_with_confidence(string dataset_path, string target_column) {
        rapidcsv::Document data(dataset_path);
        int n_rows = data.GetRowCount();
        vector<double> weighted_sum(n_rows, 0.0);
        vector<double> confidence(n_rows, 0.0);

        for (size_t i = 0; i < ensemble.size(); ++i) {
            vector<int> preds = ensemble[i]->predict(dataset_path, target_column);
            double alpha = model_weights[i];

            for (size_t j = 0; j < n_rows; ++j) {
                weighted_sum[j] += alpha * preds[j];
            }
        }

        // calculate confidence scores (normalized absolute values of weighted sums)
        double max_weight = *max_element(weighted_sum.begin(), weighted_sum.end(),
            [](double a, double b) { return abs(a) < abs(b); });

        for (size_t j = 0; j < n_rows; ++j) {
            confidence[j] = abs(weighted_sum[j]) / abs(max_weight);
        }

        // apply sign function to get final predictions
        vector<int> final_prediction(n_rows);
        transform(weighted_sum.begin(), weighted_sum.end(), final_prediction.begin(),
            [](double x) { return x >= 0 ? 1 : -1; });

        // print detailed results
        cout << "\n=== AdaBoost Detailed Predictions ===" << endl;
        cout << "Row\tWeighted Sum\tConfidence\tPrediction" << endl;
        cout << "---\t------------\t----------\t----------" << endl;

        for (size_t i = 0; i < final_prediction.size(); ++i) {
            cout << i + 1 << "\t" << fixed << setprecision(4) << weighted_sum[i]
                << "\t\t" << confidence[i] << "\t\t" << final_prediction[i] << endl;
        }
        return { final_prediction, confidence };
    }

private:
    vector<unique_ptr<WeakClassifier>> ensemble;
    vector<double> model_weights;
};


int main() {
    AdaBoost ada;
    NaiveBayes nb;
    ada.learn("drugY.csv", "Drug", nb);
    ada.predict("drugY.csv", "Drug");

    return 0;
}