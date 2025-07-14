#pragma once
// NaiveBayes.h
#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <map>
#include <string>
#include <set>
#include <cmath>
#include <vector>
#include <iostream>
#include "rapidcsv.h"
#include "WeakClassifier.h"

class NaiveBayes : public WeakClassifier {
private:
    const double pi = 3.1415926535897932;

    std::map<std::string, double> priors;
    std::map<std::string, std::map<std::string, std::map<std::string, double>>> likelihoods;
    std::map<std::string, std::map<std::string, std::pair<double, double>>> numerical_likelihoods;

    double alpha;
    std::map<std::string, double> class_counts;
    std::map<std::string, int> feature_vocabulary_sizes;

    std::map<std::string, double> calculate_occurences(std::vector<std::string> class_column);
    std::map<std::string, double> calculate_priors(std::map<std::string, double> raw_occurences, int data_size);
    double gaussian_prob(double x, double mean, double var);

public:
    NaiveBayes(double alpha = 1.0);

    void learn(std::string dataset_path, std::string target_column_name) override;
    void print_priors();
    void print_likelihoods();
    void predict(std::string dataset_path, std::string target_column_name) override;
};

#endif // NAIVE_BAYES_H
