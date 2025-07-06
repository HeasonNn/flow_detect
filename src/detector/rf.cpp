// rf_wrapper.cpp
#include "algorithm.hpp"
#include <iomanip>

void RandomForestAlgorithm::addSample(const arma::vec& x, size_t label) {
    sample_vecs.push_back(x);
    labels.push_back(label);
}


void RandomForestAlgorithm::train() {
    if (trained || sample_vecs.empty()) return;

    const size_t dim = sample_vecs[0].n_elem;
    arma::mat X(dim, sample_vecs.size());
    for (size_t i = 0; i < sample_vecs.size(); ++i)
        X.col(i) = sample_vecs[i];

    arma::Row<size_t> Y(labels);
    rf.Train(X, Y, 2);
    trained = true;
}


size_t RandomForestAlgorithm::predict(const arma::vec& x) {
    arma::Row<size_t> pred;
    rf.Classify(x, pred);
    return pred(0);
}


void RandomForestAlgorithm::printFeatures() const
{
    if (sample_vecs.empty()) {
        std::cout << "No features to print." << std::endl;
        return;
    }

    std::cout << "Feature Vectors:" << std::endl;
    std::cout << std::setw(8) << "Sample";

    for (size_t i = 0; i < sample_vecs[0].n_elem; ++i) {
        std::cout << std::setw(12) << "Field " << i + 1;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < sample_vecs.size(); ++i)
    {
        std::cout << std::setw(8) << "Sample " << i + 1;
        for (size_t j = 0; j < sample_vecs[i].n_elem; ++j) {
            std::cout << std::setw(12) << sample_vecs[i][j];
        }
        std::cout << std::endl;
    }
}