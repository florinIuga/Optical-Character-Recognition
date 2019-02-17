/**
 * Copyright 2018 Florin-Eugen Iuga
 *        
 * 
 * Project 3 SD - Optical Character Recognition on handwriting
 * 
 * Implementările metodelor din clasa decisionTree
 */

#include "./decisionTree.h"  // NOLINT(build/include)
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <numeric>  // std::accumulate


using std::string;
using std::pair;
using std::vector;
using std::unordered_map;
using std::make_shared;
using std::find;
using std::shared_ptr;
using std::mt19937;
using std::random_device;
using std::uniform_int_distribution;
using std::accumulate;

/**
 * Constructor
 * 
 * Structura unui nod din decision tree:
 * -> splitIndex = dimensiunea în funcție de care se împarte
 * -> split_value = valoarea în funcție de care se împarte
 * -> is_leaf și result sunt pentru cazul în care avem un nod frunză
 */
Node::Node() {
    is_leaf = false;
    left = nullptr;
    right = nullptr;
}

/**
 * Nodul curent devine nod de decizie
 */
void Node::make_decision_node(const int index, const int val) {
    split_index = index;
    split_value = val;
}


/**
 * 
 * 
 * Setează nodul ca fiind de tip frunză (modifică is_leaf și result).
 * 
 * @param is_single_class = true -> toate testele au aceeași clasă (result)
 * @param is_single_class = false -> se alege clasa care apare cel mai des
 */
void Node::make_leaf(const vector<vector<int>> &samples,
                     const bool is_single_class) {
    this->is_leaf = true;

    if (is_single_class) {
        this->result = samples[0][0];
        return;
    }

    vector<int> freq(10, 0);

    int max = 0;
    for (int i = 0; i < samples.size(); ++i) {
        ++freq[samples[i][0]];
    }

    for (int i = 0; i < 10; ++i) {
        if (freq[i] > max) {
            max = freq[i];
            this->result = i;
        }
    }
}

/**
 * 
 * 
 * Întoarce cea mai bună dimensiune și valoare de split dintre testele
 * primite. 
 * Cel mai bun split este cel care maximizeaza Information Gain.
 * 
 * @returnează un o pereche (split_index, split_value)
 */
pair<int, int> find_best_split(const vector<vector<int>> &samples,
                               const vector<int> &dimensions) {
    int splitIndex = -1, splitValue = -1;
    int media, nl, nr, index;
    float max = -1.0;
    float HL, HR, entropyChildren, IG;
    float entropyTarget = get_entropy(samples);

    for (int i = 0; i < dimensions.size(); ++i) {
        /**
         * Media aritmetică pe fiecare coloană al cărei index se află în
         * dimensions, pentru a face un split cât mai eficient.
         */
        media = 0;
        index = dimensions[i];
        vector<int> col = compute_unique(samples, index);
        int size = col.size();

        // Suma elementelor unice de pe coloană
        media = accumulate(col.begin(), col.end(), 0);
        // Media aritmetică
        media /= size;

        // Se realizează splitul
        auto pair = split(samples, index, media);
        nl = pair.first.size();
        nr = pair.second.size();

        // Se verifică validitatea lui și ce IG are
        if (nl && nr) {
            HL = get_entropy(pair.first);
            HR = get_entropy(pair.second);
            // Media ponderată a entropiilor copiilor
            entropyChildren = (nl * HL + nr * HR) / (nl + nr);
            IG = entropyTarget - entropyChildren;

            if (IG > max) {
                max = IG;
                splitIndex = index;
                splitValue = media;
            }
        }
    }

    return pair<int, int>(splitIndex, splitValue);
}

/**
 * 
 * 
 * Antrenează nodul curent și copii săi, dacă e nevoie.
 * 1) Verifică dacă toate testele primite au aceeași clasă (același răspuns).
 * Dacă da, acest nod devine frunza, altfel continuă algoritmul.
 * 2) Dacă nu exista niciun split valid, acest nod devine frunză. Altfel,
 * ia cel mai bun split și continuă recursiv.
 */
void Node::train(const vector<vector<int>> &samples) {
    bool is_single_class = same_class(samples);

    if (is_single_class) {
        this->make_leaf(samples, is_single_class);
    } else {
        vector<int> dim = random_dimensions(samples[0].size());
        auto p = find_best_split(samples, dim);

        if (p.first == -1 && p.second == -1){
            this->make_leaf(samples, is_single_class);
        } else {
            this->make_decision_node(p.first, p.second);

            auto children = split(samples, p.first, p.second);

            left = make_shared<Node>();
            right = make_shared<Node>();
            left->train(children.first);
            right->train(children.second);
        }
    }
}

/**
 * 
 * Funcția de prezicere dintr-un arbore de decizie
 * 
 * @returnează rezultatul prezis de către decision tree.
 */
int Node::predict(const vector<int> &image) const {
    // Dacă s-a ajuns pe o frunză, acesta este rezultatul prezis.
    if (this->is_leaf)
        return this->result;

    // Altfel, se avansează în arbore după regula de split.
    if (image[split_index - 1] <= split_value)
        return left->predict(image);
    else
        return right->predict(image);
}

/**
 * 
 * 
 * Verifică dacă testele primite ca argument au toate aceeași
 * clasă.
 * 
 * @param samples -> clasa ce trebuie prezisă pentru fiecare test se află pe
 *                  prima coloană.
 * 
 * @returnează adevărat dacă toate testele sunt de aceeași clasă, fals altfel.
 */
bool same_class(const vector<vector<int>> &samples) {
    for (int i = 1; i < samples.size(); ++i) {
        if (samples[i][0] != samples[0][0])
            return false;
    }

    return true;
}

/**
 * Funcția de calculare a entropiei
 * 
 * @returnează entropia testelor primite
 */
float get_entropy(const vector<vector<int>> &samples) {
    assert(!samples.empty());
    int size = samples.size();
    vector<int> indexes(size);

    // Am optimizat adăugarea în vector, deoarece push_back este mai lent
    for (int i = 0; i < size; i++)
        indexes[i] = i;

    return get_entropy_by_indexes(samples, indexes);
}

/**
 * 
 * 
 * Calculează entropia subsetului din setul de teste total.
 * 
 * @param index -> condiția este ca subsetul să conțină testele ai căror
 *                indecși se găsesc în vectorul index.
 * (Se consideră doar liniile din vectorul index)
 * 
 * @returnează entropia.
 */
float get_entropy_by_indexes(const vector<vector<int>> &samples,
                             const vector<int> &index) {
    vector<int> freq(10, 0);
    float entropy = 0.0;
    float total = index.size();

    for (int i = 0; i < total; ++i) {
        ++freq[ samples[ index[i] ][ 0 ] ];
    }

    for (int i = 0; i < 10; ++i) {
        if (freq[i]) {
            float Pi = freq[i] / total;
            entropy -= Pi * log2(Pi);
        }
    }

    return entropy;
}

/**
 * 
 * 
 * Elimină duplicatele care apar in setul de teste pe o coloană.
 * 
 * @param col -> coloana de pe care se elimină duplicatele.
 * 
 * @returnează un vector cu elementele unice de pe acea coloană.
 */
vector<int> compute_unique(const vector<vector<int>> &samples, const int col) {
    vector<int> uniqueValues;
    vector<int> mark(256, 0);

    for (int i = 0; i < samples.size(); ++i) {
        if (!mark[ samples[i][col] ]) {
            uniqueValues.push_back(samples[i][col]);
            ++mark[ samples[i][col] ];
        }
    }

    return uniqueValues;
}

/**
 * Funcție de split
 * 
 * @returnează cele 2 subseturi de teste obținute în urma separării în funcție
 * de split_index și split_value
 */
pair<vector<vector<int>>, vector<vector<int>>> split(
                                            const vector<vector<int>> &samples,
                                            const int split_index,
                                            const int split_value) {
    vector<vector<int>> left, right;

    auto p = get_split_as_indexes(samples, split_index, split_value);
    for (const auto &i : p.first) left.push_back(samples[i]);
    for (const auto &i : p.second) right.push_back(samples[i]);

    return pair<vector<vector<int>>, vector<vector<int>>>(left, right);
}

/**
 * 
 * 
 * Funcție de split
 * 
 * @returnează o pereche de vectori conținând indecșii sample-urilor din cele 2
 * subseturi obținute în urma separării în funcție de split_index și
 * split_value.
 */
pair<vector<int>, vector<int>> get_split_as_indexes(
                                            const vector<vector<int>> &samples,
                                            const int split_index,
                                            const int split_value) {
    vector<int> left, right;

    for (int i = 0; i < samples.size(); ++i) {
        if (samples[i][split_index] <= split_value)
            left.push_back(i);
        else
            right.push_back(i);
    }

    return make_pair(left, right);
}

/**
 * 
 * 
 * Generează dimensiuni diferite pe care să caute splitul maxim.
 * Dimensiunile găsite sunt > 0 și < size și nu se repetă.
 * 
 * @returneză un vector cu sqrt(size) dimensiuni diferite.
 */
vector<int> random_dimensions(const int size) {
    int generated;
    int count = 0;
    int dim = floor(sqrt(size));
    vector<int> rez(dim);
    vector<int> mark(size, 0);

    // Funcție de random cu distribuție foarte bună (tinde spre true random).
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> dist(1, size - 1);

    while (count < dim) {
        generated = dist(mt);
        if (!mark[generated]) {
            ++mark[generated];
            rez[count++] = generated;
        }
    }

    return rez;
}
