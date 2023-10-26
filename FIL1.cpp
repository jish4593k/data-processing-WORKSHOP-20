#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <algorithm>
#include "Eigen/Dense"
#include "GMM.h"

using namespace Eigen;

// Define a structure to represent hero data
struct HeroData {
    double MaxHP, InitialHP, MaxMana, MaxPhysicalAttack, InitialPhysicalAttack, MaxPhysicalDefense, InitialPhysicalDefense;
    double MaxHPRegen, MaxManaRegen, InitialManaRegen, MaxAttackSpeed, AttackRange;
    int Cluster;

    HeroData() : Cluster(0) {}
};

int main() {
    // Load hero data (replace 'heroes7.csv' with your data file)
    std::ifstream file("heroes7.csv");
    std::vector<HeroData> heroData;

    std::string line;
    std::getline(file, line); // Skip the header

    while (std::getline(file, line)) {
        HeroData hero;
        std::istringstream iss(line);
        std::string token;

        std::getline(iss, token, ',');
        hero.MaxHP = std::stod(token);
        std::getline(iss, token, ',');
        hero.InitialHP = std::stod(token);
        std::getline(iss, token, ',');
        hero.MaxMana = std::stod(token);
        // ... Parse other attributes similarly

        heroData.push_back(hero);
    }

    // Data Preprocessing
    // Normalize data (Z-Score)
    MatrixXd data(heroData.size(), 12);
    for (size_t i = 0; i < heroData.size(); i++) {
        data(i, 0) = (heroData[i].MaxHP - mean_MaxHP) / std_MaxHP;
        // ... Normalize other attributes similarly
    }

    // Clustering using Gaussian Mixture Model (GMM)
    int numClusters = 10;
    GMM gmm(numClusters);
    gmm.train(data);

    // Assign clusters to heroes
    for (size_t i = 0; i < heroData.size(); i++) {
        heroData[i].Cluster = gmm.predict(data.row(i));
    }

    // Cluster Statistics
    std::map<int, int> clusterCounts;
    for (const HeroData& hero : heroData) {
        clusterCounts[hero.Cluster]++;
    }

    // Display cluster statistics
    for (const auto& entry : clusterCounts) {
        std::cout << "Cluster " << entry.first << ": " << entry.second << " heroes" << std::endl;
    }

    // Calculate and display the Calinski-Harabaz score
    // You'll need to implement this calculation based on your needs

    return 0;
}
