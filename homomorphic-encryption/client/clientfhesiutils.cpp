//
// Created by George Sakellariou on 29/09/2017.
//

#include "clientfhesiutils.h"

using namespace std;

bool LoadDataPolyX(vector<ZZ_pX> &rawData, vector<ZZ_p> &labels, unsigned &dim,
                   const string &filename, FHEcontext &context) {
  int label, n, n_clusters;
  long phim = context.zMstar.phiM();
  ifstream fin;
  fin.open(filename);
  if (!fin) {
    cout << "Unable to read data file." << endl;
    return false;
  }

  rawData.clear();
  labels.clear();

  ZZ p = context.ModulusP();
  fin >> dim >> n >> n_clusters;  // TODO

  ZZ_pX data;
  data.SetMaxLength(phim);
  long temp;
  for (int i = 0; i < n; i++) {
    for (unsigned j = 0; j < dim; j++) {
      fin >> temp;
      SetCoeff(data, j, temp);
    }
    label = 0;  // fin >> label; // TODO: skip inputting labels
    rawData.push_back(data);
    labels.push_back(to_ZZ_p(label));
  }

  return true;
}

bool LoadDataVecPolyX(vector<vector<ZZ_pX>> &rawData, vector<ZZ_p> &labels,
                      unsigned &dim, const string &filename,
                      FHEcontext &context,
                      vector<vector<uint32_t>> &rawDatatoInt) {
  int label, n, n_clusters;
  long phim = context.zMstar.phiM();
  ifstream fin;
  fin.open(filename);
  if (!fin) {
    cout << "Unable to read data file." << endl;
    return false;
  }

  rawData.clear();
  labels.clear();

  ZZ p = context.ModulusP();
  fin >> dim >> n >> n_clusters;  // TODO

  ZZ_pX data;
  data.SetMaxLength(1);
  uint32_t coeftemp;
  long temp;
  for (int i = 0; i < n; i++) {
    vector<ZZ_pX> point;
    vector<uint32_t> pointToInt;
    for (unsigned j = 0; j < dim; j++) {
      fin >> temp;
      coeftemp = static_cast<uint32_t>(temp);
      SetCoeff(data, 0, temp);
      point.push_back(data);
      pointToInt.push_back(coeftemp);
    }
    label = 0;  // fin >> label; // TODO: skip inputting labels
    rawData.push_back(point);
    labels.push_back(to_ZZ_p(label));
    rawDatatoInt.push_back(pointToInt);
  }

  return true;
}

vector<Ciphertext> EncryptVector(const vector<ZZ_pX> &point,
                                 const FHEcontext &fhEcontext,
                                 const FHESIPubKey &fhesiPubKey) {
  unsigned long dimension = point.size();
  vector<Ciphertext> encrypted_vector;
  for (unsigned i = 0; i < dimension; i++) {
    Plaintext coefficient(fhEcontext, point[i]);
    Ciphertext encrypted_coefficient(fhesiPubKey);
    fhesiPubKey.Encrypt(encrypted_coefficient, coefficient);
    encrypted_vector.push_back(encrypted_coefficient);
  }
  return encrypted_vector;
}

vector<ZZ_pX> DecryptVector(const vector<Ciphertext> &cpoint,
                            const FHESISecKey &fhesiSecKey) {
  unsigned long dimension = cpoint.size();
  vector<ZZ_pX> decrypted_vector;
  for (int i = 0; i < dimension; ++i) {
    Plaintext coefficient;
    fhesiSecKey.Decrypt(coefficient, cpoint[i]);
    decrypted_vector.push_back(coefficient.message);
  }
  return decrypted_vector;
}

void timeCalulator(const clock_t &c_start,
                   const chrono::high_resolution_clock::time_point &t_start) {
  std::clock_t c_end = std::clock();
  auto t_end = std::chrono::high_resolution_clock::now();
  std::cout << fixed << setprecision(2)
            << "CPU time used: " << 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC
            << " ms\n"
            << "Wall clock time passed: "
            << chrono::duration<double, milli>(t_end - t_start).count() << " ms"
            << endl;
}

long combine(long a, long b) {
  int times = 1;
  while (times <= b) times *= 10;
  return a * times + b;
}
