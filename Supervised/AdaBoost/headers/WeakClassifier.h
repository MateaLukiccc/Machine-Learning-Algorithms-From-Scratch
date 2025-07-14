#pragma once
#include <string>

class WeakClassifier {
public:
	virtual void learn(std::string dataset_path, std::string target_column_name) = 0;
	virtual void predict(std::string dataset_path, std::string target_column_name) = 0;
	virtual ~WeakClassifier() = default;
};