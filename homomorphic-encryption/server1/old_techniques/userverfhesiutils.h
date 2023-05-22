//
// Created by George Sakellariou on 29/09/2017.
//

#ifndef USERVER_USERVERFHESIUTILS_H
#define USERVER_USERVERFHESIUTILS_H

#include <bitset>
#include <fstream>
#include <iostream>
#include <string>

#include "Ciphertext.h"
#include "FHE-SI.h"
#include "FHEContext.h"
#include "Matrix.h"
#include "ZZ_pX.h"
#include "chrono"
#include "ctime"
#include "iomanip"

Ciphertext FHE_HM(Ciphertext &c1, Ciphertext &c2);
void timeCalulator(const clock_t &c_start,
                   const chrono::high_resolution_clock::time_point &t_start);
Ciphertext euclideanDistance(vector<Ciphertext> &cpoint1,
                             vector<Ciphertext> &cpoint2,
                             KeySwitchSI &keySwitchSI);
Ciphertext euclideanDistanceP(Ciphertext &c1, Ciphertext &c2,
                              KeySwitchSI &keySwitchSI);

template <typename T>
void print(const T &message) {
  // std::cout << message << std::endl; // force quiet
}

#endif