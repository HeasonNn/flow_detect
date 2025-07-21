#pragma once

#include <mlpack/core.hpp>
#include <mlpack/core/math/random.hpp>


using std::unique_ptr;
using std::make_unique;
using std::vector;

constexpr double EULER_MASCHERONI = 0.57721566490153286060;


static inline double c_factor(int n) {
    if (n <= 1) return 0.0;
    return 2.0 * (std::log(n - 1.0) + EULER_MASCHERONI) - 2.0 * (n - 1.0) / n;
}


struct Node {
    int featureIndex   = -1;
    double splitValue  = 0.0;
    int size           = 0;

    unique_ptr<Node> left;
    unique_ptr<Node> right;

    bool IsLeaf() const { return !left && !right; }
};


class IsolationTree {
public:
    explicit IsolationTree(size_t maxDepth) : maxDepth(maxDepth) {}

    unique_ptr<Node> Fit(const arma::mat& data, int depth = 0) {
        if (data.n_cols <= 1 || data.n_rows == 0 || depth >= maxDepth)
            return MakeLeaf(data.n_cols);

        int featureIndex = mlpack::RandInt(0, static_cast<int>(data.n_rows));
        double minVal = data.row(featureIndex).min();
        double maxVal = data.row(featureIndex).max();

        if (minVal == maxVal)
            return MakeLeaf(data.n_cols);

        double split = mlpack::Random(minVal, maxVal);
        arma::uvec leftIndices  = arma::find(data.row(featureIndex) <  split);
        arma::uvec rightIndices = arma::find(data.row(featureIndex) >= split);

        if (leftIndices.n_elem == 0 || rightIndices.n_elem == 0)
            return MakeLeaf(data.n_cols);

        arma::mat leftData  = data.cols(leftIndices);
        arma::mat rightData = data.cols(rightIndices);

        auto node = make_unique<Node>();
        node->featureIndex = featureIndex;
        node->splitValue   = split;
        node->size         = data.n_cols;
        node->left  = Fit(leftData,  depth + 1);
        node->right = Fit(rightData, depth + 1);

        return node;
    }

    static double PathLength(const arma::vec& point,
                             const unique_ptr<Node>& node,
                             size_t depth = 0) {
        if (!node)
            return depth;
        if (node->IsLeaf())
            return depth + c_factor(node->size);
        if (point[node->featureIndex] < node->splitValue)
            return PathLength(point, node->left,  depth + 1);
        else
            return PathLength(point, node->right, depth + 1);
    }

private:
    int maxDepth;

    unique_ptr<Node> MakeLeaf(int size) {
        auto leaf = make_unique<Node>();
        leaf->size = size;
        return leaf;
    }
};


class IsolationForest {
public:
    IsolationForest(int random_seed, size_t nTrees, size_t sampleSize, size_t maxDepth)
        : random_seed_(random_seed), 
          nTrees(nTrees), 
          sampleSize(sampleSize), 
          maxDepth(maxDepth) {}

    void Fit(const arma::mat& data) {
        arma::arma_rng::set_seed(random_seed_);

        trees.clear();
        vector<unique_ptr<Node>> local_trees(nTrees);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(nTrees); ++i) {
            arma::uvec indices = arma::randperm(data.n_cols, sampleSize);
            arma::mat sample = data.cols(indices);
            IsolationTree tree(maxDepth);
            local_trees[i] = tree.Fit(sample);
        }

        trees = std::move(local_trees);
    }

    double AnomalyScore(const arma::vec& point) const {
        double pathLengthSum = 0.0;

        #pragma omp parallel for reduction(+:pathLengthSum)
        for (int i = 0; i < static_cast<int>(trees.size()); ++i) {
            pathLengthSum += IsolationTree::PathLength(point, trees[i]);
        }

        double avgPathLength = pathLengthSum / nTrees;
        double cn = c_factor(sampleSize);
        return std::pow(2.0, -avgPathLength / cn);
    }

private:
    int random_seed_;
    size_t nTrees;
    size_t sampleSize;
    size_t maxDepth;
    vector<unique_ptr<Node>> trees;
};
