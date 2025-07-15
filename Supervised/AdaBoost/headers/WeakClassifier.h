#pragma once
#include <string>
#include <vector>

class WeakClassifier {
public:
	virtual void learn(std::string dataset_path, std::string target_column_name, std::vector<double> instance_weights) = 0;
	virtual std::vector<int> predict(std::string dataset_path, std::string target_column_name) = 0;
	virtual std::unique_ptr<WeakClassifier> clone() const = 0;
	virtual ~WeakClassifier() = default;
};