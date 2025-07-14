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

using namespace std;

class AdaBoost {
public:
    void learn(string dataset_path, string target_column) {
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
        vector<double> alphas(ensamble_size, 0.0);

        // TODO: preprocess to get only -1 and 1
        vector<int> labels = data.GetColumn<int>(target_column);

        // fit each classifier
        for (int classifier = 0; classifier < ensamble_size; classifier++) {
            // need to make it possible to choose classifier
            unique_ptr<WeakClassifier> weak_classifier = make_unique<NaiveBayes>();
            
            // TODO: change implementation to support weights for instances
            weak_classifier->learn(dataset_path, target_column);

            // TODO: need to change implementation to get back predictions
            weak_classifier->predict(dataset_path, target_column);
            vector<int> predictions(n_rows, 0);

            double weighted_error = 0.0;
            for (size_t i = 0; i < n_rows; ++i) {
                if (predictions[i] != labels[i]) {
                    weighted_error += D_t[i];
                }
            }

            // Compute model weight (alpha)
            double alpha = 0.5 * log((1 - weighted_error) / weighted_error);

            // Store model and alpha
            ensemble.push_back(move(weak_classifier));
            alphas.push_back(alpha);


            // Update instance weights
            for (size_t i = 0; i < n_rows; ++i) {
                D_t[i] *= exp(-alpha * predictions[i] * labels[i]);
            }

            // Normalize weights
            double sum_D = accumulate(D_t.begin(), D_t.end(), 0.0);
            for (double& d : D_t) d /= sum_D;
        }
        // Store results
        this->ensemble = move(ensemble);
        this->model_weights = move(model_weights);
    }

    vector<int> predict(string dataset_path, string target_column) {
        rapidcsv::Document data(dataset_path);
        int n_rows = data.GetRowCount();
        vector<int> final_prediction(n_rows, 0);

        for (size_t i = 0; i < ensemble.size(); ++i) {
            vector<int> preds = { 0, 0, 0 };
            // TODO: predicts doesnt return predictions    
            ensemble[i]->predict(dataset_path, target_column);
            double alpha = model_weights[i];

            for (size_t j = 0; j < n_rows; ++j) {
                final_prediction[j] += alpha * preds[j];
            }
        }

        // Apply sign function
        for (int& pred : final_prediction) {
            pred = (pred >= 0) ? 1 : -1;
        }
        return final_prediction;
    }

private:
    vector<unique_ptr<WeakClassifier>> ensemble;
    vector<double> model_weights;
};


int main()
{
    // transform target to 1 -1
    NaiveBayes nb;
    nb.predict("prehlada_novi.csv", "Prehlada");

    AdaBoost ada;
    ada.learn("drugY.csv", "Drug");

    return 0;
}
