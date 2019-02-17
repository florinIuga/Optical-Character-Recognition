/**
 * Copyright 2018 Florin-Eugen Iuga
 * 
 * Tema 3 SD - Optical Character Recognition on handwriting
 * 
 * Implementările metodelor din clasa randomForest
 */

#include "randomForest.h"
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "decisionTree.h"

using std::vector;
using std::pair;
using std::string;
using std::mt19937;
using std::max_element;
using std::distance;
using std::random_device;
using std::uniform_int_distribution;

/**
 * 
 * 
 * Funcția de selecție a unor teste aleatoare.
 * 
 * @returnează un vector de mărime num_to_return cu linii random și diferite
 * din samples.
 */
vector<vector<int>> get_random_samples(const vector<vector<int>> &samples,
                                       int num_to_return) {
    int size = samples.size();
    int i = 0;
    vector<int> mark(size, 0);
    vector<vector<int>> ret(num_to_return, vector<int>());

    // Funcție de random cu distribuție foarte bună (tinde spre true random).
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(0, size - 1);

    while (i < num_to_return) {
        int index = dist(mt);

        if (!mark[index]) {
            ret[i++] = samples[index];
            ++mark[index];
        }
    }

    return ret;
}

RandomForest::RandomForest(int num_trees, const vector<vector<int>> &samples)
    : num_trees(num_trees), images(samples) {}

/**
 * Alocă pentru fiecare Tree câte n / num_trees, unde n e numărul total de
 * teste de training, apoi antrenează fiecare tree cu testele alese.
 */
void RandomForest::build() {
    assert(!images.empty());
    vector<vector<int>> random_samples;

    int data_size = images.size() / num_trees;

    for (int i = 0; i < num_trees; i++) {
        random_samples = get_random_samples(images, data_size);

        // Construiește un Tree nou și îl antrenează
        trees.push_back(Node());
        trees[trees.size() - 1].train(random_samples);
    }
}

/**
 * 
 * 
 * Află cea mai probabilă prezicere pentru testul din argument, interogând
 * fiecare Tree și se va considera răspunsul final ca fiind cel majoritar.
 * 
 * @returnează predicția făcută.
 */
int RandomForest::predict(const vector<int> &image) {
    int max = 0;
    int pos;
    vector<int> freq(10, 0);
    vector<Node>::iterator it;

    for (it = trees.begin(); it != trees.end(); ++it) {
        int predicted = it->predict(image);
        ++freq[predicted];

        /**
         * Optimizare de timp (dacă mai mult de jumătate din teste au aceeași
         * predicție, atunci algoritmul se oprește.
         */
        if (freq[predicted] > num_trees / 2)
            return predicted;
    }

    for (int i = 0; i < 10; ++i) {
        if (freq[i] > max) {
            max = freq[i];
            pos = i;
        }
    }

    return pos;
}
