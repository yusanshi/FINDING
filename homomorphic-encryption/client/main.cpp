//
// Created by george on 8/11/2017.
//

#include <chrono>
#include <ctime>
#include <iomanip>

#include "KClientT2V1.h"

int main(int argc, char **argv) {
  unsigned p = 2027;  // 1487;//1487
  unsigned g = 7;
  unsigned logQ = 55;  // 4

  KClientT2V1 client(p, g, logQ, argc > 1 ? argv[1] : "../sample.dat",
                     "127.0.0.1", 5001, "127.0.0.1", 5002, 3, true);
}
