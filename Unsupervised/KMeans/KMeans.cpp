#include <iostream>
#include <string>
#include "rapidcsv.h"
#include <vector>
#include <random>
#include <set>
#include <cmath>
#include <limits>

using namespace std;

class KMeans {
private:
    int K;
    vector<vector<double>> centroids;
    vector<double> weights;

    void standardize_column(rapidcsv::Document& data, const string& column_name, int col_idx) {
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

        // Standardize: (x - mean) / std_dev
        for (double& v : column_data) {
            v = (v - mean) / std_dev;
        }

        data.SetColumn(column_name, column_data);
    }

    void print_centroids(rapidcsv::Document data) {
        cout << "\nNormalized centroids:" << endl;
        for (int c = 0; c < K; c++) {
            cout << "Centroid " << c << ": ";
            for (int j = 0; j < data.GetColumnCount(); j++) {
                cout << centroids[c][j] << " ";
            }
            cout << endl;
        }
    }

    void print_assigned_cluster(rapidcsv::Document data, vector<int> assignments, int c_id) {
        if (c_id >= K) {
            cout << "\nWrong cluster id, this cluster does not exist\n";
            return;
        }
        cout << "\nPoints assigned to cluster " << c_id << ":" << endl;
        for (int i = 0; i < data.GetRowCount(); i++) {
            if (assignments[i] == c_id) {
                cout << "Row " << i << ": ";
                for (int j = 0; j < data.GetColumnCount(); j++) {
                    cout << data.GetCell<double>(j, i) << " ";
                }
                cout << endl;
            }
        }
    }

    void check_cluster_quality(rapidcsv::Document& data, const vector<int>& assignments) {
        vector<double> avg_distances(K, 0.0);

        // Compute average distance for each cluster
        for (int c = 0; c < K; ++c) {
            vector<int> cluster_points;
            for (int i = 0; i < assignments.size(); ++i) {
                if (assignments[i] == c) cluster_points.push_back(i);
            }

            if (cluster_points.empty()) {
                cout << "UPOZORENJE: Klaster " << c << " je prazan!\n";
                continue;
            }

            double sum_distances = 0;
            for (int idx : cluster_points) {
                double distance = 0;
                for (int j = 0; j < data.GetColumnCount(); ++j) {
                    double diff = data.GetCell<double>(j, idx) - centroids[c][j];
                    distance += diff * diff * weights[j];
                }
                sum_distances += sqrt(distance);
            }

            avg_distances[c] = sum_distances / cluster_points.size();
        }

        // Compute average of average distances
        double total = 0;
        int count = 0;
        for (double d : avg_distances) {
            if (d > 0) {
                total += d;
                count++;
            }
        }
        double global_avg = (count > 0) ? (total / count) : 0.0;
        double max_allowed = global_avg * 1.5;

        // Report clusters that exceed threshold
        for (int c = 0; c < K; ++c) {
            if (avg_distances[c] > max_allowed) {
                cout << "UPOZORENJE: Klaster " << c << " ima visoku prose?nu udaljenost od centroida ("
                    << avg_distances[c] << ", o?ekivano <= " << max_allowed << ").\n";
            }
        }
    }

    void check_centroid_similarity() {
        vector<double> distances;

        // Compute all pairwise centroid distances
        for (int i = 0; i < K; ++i) {
            for (int j = i + 1; j < K; ++j) {
                double distance = 0;
                for (int d = 0; d < centroids[i].size(); ++d) {
                    double diff = centroids[i][d] - centroids[j][d];
                    distance += diff * diff * weights[d];
                }
                distance = sqrt(distance);
                distances.push_back(distance);
            }
        }

        if (distances.empty()) return;

        // Compute average distance between centroids
        double sum = 0.0;
        for (double d : distances) sum += d;
        double avg_distance = sum / distances.size();

        double min_acceptable = avg_distance * 0.8;

        // Compare each pair again and print warning
        int index = 0;
        for (int i = 0; i < K; ++i) {
            for (int j = i + 1; j < K; ++j) {
                double d = distances[index++];
                if (d < min_acceptable) {
                    cout << "UPOZORENJE: Centroidi klastera " << i << " i " << j
                        << " su previ�e blizu (udaljenost: " << d
                        << ", o?ekivano >= " << min_acceptable << ").\n";
                }
            }
        }
    }

    double silhouette_score(const rapidcsv::Document& data, const vector<int>& assignments) {
        int n = data.GetRowCount();
        int d = data.GetColumnCount();
        double total_score = 0.0;

        for (int i = 0; i < n; ++i) {
            int cluster_i = assignments[i];
            vector<double> point_i = data.GetRow<double>(i);

            // Calculate a(i)
            double a = 0.0;
            int a_count = 0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (assignments[j] == cluster_i) {
                    double dist = 0.0;
                    vector<double> point_j = data.GetRow<double>(j);
                    for (int k = 0; k < d; ++k)
                        dist += pow(point_i[k] - point_j[k], 2) * weights[k];
                    a += sqrt(dist);
                    a_count++;
                }
            }
            a = (a_count > 0) ? (a / a_count) : 0;

            // Calculate b(i)
            double b = numeric_limits<double>::max();
            for (int c = 0; c < K; ++c) {
                if (c == cluster_i) continue;
                double dist_sum = 0.0;
                int count = 0;
                for (int j = 0; j < n; ++j) {
                    if (assignments[j] == c) {
                        double dist = 0.0;
                        vector<double> point_j = data.GetRow<double>(j);
                        for (int k = 0; k < d; ++k)
                            dist += pow(point_i[k] - point_j[k], 2) * weights[k];
                        dist_sum += sqrt(dist);
                        count++;
                    }
                }
                if (count > 0) {
                    double avg = dist_sum / count;
                    b = min(b, avg);
                }
            }

            double s = (a == 0 && b == 0) ? 0 : (b - a) / max(a, b);
            total_score += s;
        }

        return total_score / n;
    }

    struct LearnResult {
        double quality;
        std::vector<int> assignments;
    };

public:
    KMeans(int K, vector<double> weights) : K(K), weights(weights) {}
    KMeans(int K) : K(K) {}

    LearnResult learn(string dataset_path, bool verbose = false) {
        rapidcsv::Document data(dataset_path);
        vector<string> column_names(data.GetColumnNames());

        // check if weights are initialized
        if (weights.empty()) {
            weights = vector<double>(column_names.size(), 1);
        }
        else {
            if (weights.size() != data.GetColumnCount()) {
                cout << "Number of weights should be equal to the number of columns but its " << weights.size() << " and number of columns is " << data.GetColumnCount();
            }

            double weight_sum = 0;
            for (int i = 0; i < weights.size(); i++) {
                weight_sum += weights[i];
            }
            if (weight_sum != 1 && weight_sum != weights.size()) {
                cout << "Sum of all weights for columns should be 1 but its " << weight_sum;
                return LearnResult();
            }
        }

        // standardize data
        for (int i = 0; i < column_names.size(); i++) {
            standardize_column(data, column_names[i], i);
        }

        // Initialize centroids
        random_device rd;
        mt19937 gen(rd());

        centroids.clear();
        centroids.resize(K, vector<double>(data.GetColumnCount()));

        // Step 1: Choose the first centroid randomly
        uniform_int_distribution<> distribution(0, data.GetRowCount() - 1);
        int first_idx = distribution(gen);
        for (int j = 0; j < data.GetColumnCount(); ++j) {
            centroids[0][j] = data.GetCell<double>(j, first_idx);
        }

        // Keep track of used indices to avoid duplicates
        set<int> used_indices;
        used_indices.insert(first_idx);

        // Step 2: Choose remaining K-1 centroids based on minimum distance
        for (int c = 1; c < K; ++c) {
            vector<double> min_distances(data.GetRowCount(), numeric_limits<double>::max());

            // For each point, find minimum distance to any existing centroid
            for (int i = 0; i < data.GetRowCount(); i++) {
                if (used_indices.count(i)) continue; // Skip already used points

                double min_dist = numeric_limits<double>::max();
                vector<double> point = data.GetRow<double>(i);

                // Find minimum distance to any existing centroid
                for (int existing_c = 0; existing_c < c; existing_c++) {
                    double dist = 0.0;
                    for (int j = 0; j < data.GetColumnCount(); j++) {
                        double diff = point[j] - centroids[existing_c][j];
                        dist += diff * diff;
                    }
                    min_dist = min(min_dist, dist);
                }
                min_distances[i] = min_dist;
            }

            // Find the point with maximum minimum distance (furthest from all centroids)
            int best_idx = -1;
            double best_distance = -1;

            for (int i = 0; i < data.GetRowCount(); i++) {
                if (used_indices.count(i)) continue; // Skip already used points

                if (min_distances[i] > best_distance) {
                    best_distance = min_distances[i];
                    best_idx = i;
                }
            }

            // Add the best point as new centroid
            if (best_idx != -1) {
                for (int j = 0; j < data.GetColumnCount(); j++) {
                    centroids[c][j] = data.GetCell<double>(j, best_idx);
                }
                used_indices.insert(best_idx);
            }
        }

        // initial values
        vector<int> assignments(data.GetRowCount(), 0);
        double old_quality = numeric_limits<double>::max();

        // start iterating
        for (int iteration = 0; iteration < 50; iteration++) {
            vector<double> quality(K, 0);

            // Assignment step: assign each point to nearest centroid
            for (int i = 0; i < data.GetRowCount(); i++) {
                vector<double> distances(K, 0);

                // Calculate distance to each centroid
                for (int c = 0; c < K; c++) {
                    double distance = 0;
                    for (int j = 0; j < data.GetColumnCount(); j++) {
                        double diff = data.GetCell<double>(j, i) - centroids[c][j];
                        distance += diff * diff * weights[j];
                    }
                    distances[c] = distance;
                }

                // Find closest centroid
                double minimum_value = numeric_limits<double>::max();
                int minimum_index = 0;
                for (int c = 0; c < K; c++) {
                    if (minimum_value > distances[c]) {
                        minimum_value = distances[c];
                        minimum_index = c;
                    }
                }
                assignments[i] = minimum_index;
            }

            // Update centroids step
            for (int c = 0; c < K; c++) {
                // Find all points assigned to this centroid
                vector<int> subset_indices;
                for (int i = 0; i < data.GetRowCount(); i++) {
                    if (assignments[i] == c) {
                        subset_indices.push_back(i);
                    }
                }

                // Calculate mean for each feature
                for (int j = 0; j < data.GetColumnCount(); j++) {
                    double sum = 0;
                    for (int idx : subset_indices) {
                        sum += data.GetCell<double>(j, idx);
                    }
                    centroids[c][j] = sum / subset_indices.size();
                }

                // Calculate quality (original variance calculation)
                double total_variance = 0;
                for (int j = 0; j < data.GetColumnCount(); j++) {
                    double mean_val = centroids[c][j];
                    double variance = 0;
                    for (int idx : subset_indices) {
                        double diff = data.GetCell<double>(j, idx) - mean_val;
                        variance += diff * diff;
                    }
                    variance /= subset_indices.size(); // population variance
                    total_variance += variance;
                }
                quality[c] = total_variance * subset_indices.size();
            }

            // Calculate total quality
            double total_quality = 0;
            for (double q : quality) {
                total_quality += q;
            }

            cout << iteration << " " << total_quality << endl;

            if (old_quality == total_quality) {
                break;
            }
            old_quality = total_quality;
        }

        if (verbose) {
            print_centroids(data);
            print_assigned_cluster(data, assignments, 2);
        }

        LearnResult result;
        result.quality = old_quality;
        result.assignments = assignments;
        return result;
    }

    void learn_multiple_runs(string dataset_path, int runs = 10) {
        double best_quality = numeric_limits<double>::max();
        vector<vector<double>> best_centroids;
        vector<int> best_assignments;

        for (int i = 0; i < runs; ++i) {
            cout << "\n--- Run " << i + 1 << " ---\n";
            LearnResult result = learn(dataset_path);
            double total_quality = result.quality;
            vector<int> assignments = result.assignments;
            if (total_quality < best_quality) {
                best_quality = total_quality;
                best_centroids = centroids;
                best_assignments = assignments;
            }
        }

        rapidcsv::Document data(dataset_path);
        centroids = best_centroids;
        cout << "\nNajbolji rezultat posle " << runs << " poku�aja. Total quality: " << best_quality << endl;
        print_centroids(data);
        print_assigned_cluster(data, best_assignments, 2);
        check_cluster_quality(data, best_assignments);
        check_centroid_similarity();
    }

    int find_best_k(string dataset_path, int k_min = 2, int k_max = 10) {
        rapidcsv::Document data(dataset_path);
        vector<string> column_names = data.GetColumnNames();

        // Standardizuj podatke
        for (int i = 0; i < column_names.size(); i++) {
            standardize_column(data, column_names[i], i);
        }

        double best_score = -1.0;
        int best_k = k_min;
        vector<double> original_weights = weights;
        if (weights.empty()) weights = vector<double>(data.GetColumnCount(), 1.0);

        for (int k = k_min; k <= k_max; ++k) {
            cout << "\nTesting K = " << k << "...\n";
            KMeans km(k, weights);
            LearnResult res = km.learn(dataset_path);
            double score = km.silhouette_score(data, res.assignments);
            cout << "Silhouette score for K = " << k << ": " << score << endl;

            if (score > best_score) {
                best_score = score;
                best_k = k;
            }
        }

        cout << "\nBest K according to silhouette score: " << best_k << " (Score = " << best_score << ")\n";
        weights = original_weights; // restore
        return best_k;
    }
};

int main() {
    KMeans kmeans(3);
    kmeans.learn_multiple_runs("boston.csv");
    cout << "K-Means implementation ready!" << endl;

    int optimal_k = kmeans.find_best_k("boston.csv", 2, 8);
    KMeans final_model(optimal_k);
    final_model.learn_multiple_runs("boston.csv");
    cout << "K-Means with optimal K = " << optimal_k << " completed!" << endl;
    return 0;
}