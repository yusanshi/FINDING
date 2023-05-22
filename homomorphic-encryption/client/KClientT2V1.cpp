//
// Created by george on 16/11/2017.
//

#include "KClientT2V1.h"

KClientT2V1::KClientT2V1(unsigned p, unsigned g, unsigned logQ,
                         const string &data, const string &u_serverIP,
                         unsigned u_serverPort, const string &t_serverIP,
                         unsigned t_serverPort, unsigned k, bool verbose) {
  this->k = k;
  this->active = true;
  this->verbose = verbose;
  this->u_serverIP = u_serverIP;
  this->u_serverPort = u_serverPort;
  this->t_serverIP = t_serverIP;
  this->t_serverPort = t_serverPort;
  this->client_p = p;
  this->client_g = g;
  this->client_logQ = logQ;
  print("K-CLIENT V2");
  FHEcontext context(this->client_p - 1, this->client_logQ, this->client_p,
                     this->client_g);
  activeContext = &context;
  this->client_context = &context;
  context.SetUpSIContext();
  FHESISecKey fhesiSecKey1(context);
  FHESIPubKey fhesiPubKey1(fhesiSecKey1);
  KeySwitchSI keySwitchSI1(fhesiSecKey1);
  FHESISecKey fhesiSecKeyT(context);
  KeySwitchSI keySwitchSIT(fhesiSecKey1, fhesiSecKeyT);
  this->fhesiSecKey = &fhesiSecKey1;
  this->fhesiPubKey = &fhesiPubKey1;
  this->keySwitchSI = &keySwitchSI1;
  this->fhesiSecKeyT = &fhesiSecKeyT;
  this->keySwitchSIT = &keySwitchSIT;
  print(context);
  // print(*this->fhesiPubKey);
  // print(*this->fhesiSecKey);
  // print(*this->keySwitchSI);
  // print(*this->fhesiSecKeyT);
  // print(*this->keySwitchSIT);
  this->connectToTServer();
  this->sendEncryptionParamToTServer();
  this->connectToUServer();
  this->sendEncryptionParamToUServer();
  LoadDataPolyX(this->loadeddata, this->labels, this->dim, data,
                *this->client_context);
  this->createStruct();
  this->sendEncryptedDataToUServer();
  while (this->active) {
    string message = this->receiveMessage(this->u_serverSocket, 4);
    if (message == "U-RE") {
      this->receiveResult();
      this->active = false;
    } else if (message == "U-NC") {
      this->calculateCentroid(this->u_serverSocket);
    } else {
      perror("ERROR IN PROTOCOL INITIALIZATION");
      return;
    }
  }
}

void KClientT2V1::connectToTServer() {
  struct sockaddr_in t_server_address;
  if (this->t_serverSocket == -1) {
    this->t_serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (this->t_serverSocket < 0) {
      perror("ERROR ON TSERVER SOCKET CREATION");
      exit(1);
    } else {
      string message =
          "Socket for TServer created successfully. File descriptor: " +
          to_string(this->t_serverSocket);
      print(message);
    }
  }
  t_server_address.sin_addr.s_addr = inet_addr(this->t_serverIP.c_str());
  t_server_address.sin_family = AF_INET;
  t_server_address.sin_port = htons(static_cast<uint16_t>(this->t_serverPort));

  if (connect(this->t_serverSocket, (struct sockaddr *)&t_server_address,
              sizeof(t_server_address)) < 0) {
    perror("ERROR. CONNECTION FAILED TO TSERVER");

  } else {
    print("KClientT2V1 CONNECTED TO TSERVER");
  }
}

void KClientT2V1::connectToUServer() {
  struct sockaddr_in u_server_address;
  if (this->u_serverSocket == -1) {
    this->u_serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (this->u_serverSocket < 0) {
      perror("ERROR ON USERVER SOCKET CREATION");
      exit(1);
    } else {
      string message =
          "Socket for UServer created successfully. File descriptor: " +
          to_string(this->u_serverSocket);
      print(message);
    }
  }
  u_server_address.sin_addr.s_addr = inet_addr(this->u_serverIP.c_str());
  u_server_address.sin_family = AF_INET;
  u_server_address.sin_port = htons(static_cast<uint16_t>(this->u_serverPort));

  if (connect(this->u_serverSocket, (struct sockaddr *)&u_server_address,
              sizeof(u_server_address)) < 0) {
    perror("ERROR. CONNECTION FAILED TO USERVER");

  } else {
    print("KClientT2V1 CONNECTED TO USERVER");
  }
}

bool KClientT2V1::sendMessage(string message, int socket) {
  if (send(socket, message.c_str(), strlen(message.c_str()), 0) < 0) {
    perror("SEND FAILED.");
    return false;
  } else {
    this->log(socket, "<--- " + message);
    return true;
  }
}

bool KClientT2V1::sendStream(ifstream data, int socket) {
  uint32_t CHUNK_SIZE = 10000;
  streampos begin, end;
  begin = data.tellg();
  data.seekg(0, ios::end);
  end = data.tellg();
  streampos size = end - begin;
  uint32_t sizek;
  sizek = static_cast<uint32_t>(size);
  data.seekg(0, std::ios::beg);
  auto *memblock = new char[sizek];
  data.read(memblock, sizek);
  data.close();
  htonl(sizek);
  if (0 > send(socket, &sizek, sizeof(uint32_t), 0)) {
    perror("SEND FAILED.");
    return false;
  } else {
    this->log(socket, "<--- " + to_string(sizek));
    if (this->receiveMessage(socket, 7) == "SIZE-OK") {
      auto *buffer = new char[CHUNK_SIZE];
      uint32_t beginmem = 0;
      uint32_t endmem = 0;
      uint32_t num_of_blocks = sizek / CHUNK_SIZE;
      uint32_t rounds = 0;
      while (rounds <= num_of_blocks) {
        if (rounds == num_of_blocks) {
          uint32_t rest = sizek - (num_of_blocks)*CHUNK_SIZE;
          endmem += rest;
          copy(memblock + beginmem, memblock + endmem, buffer);
          ssize_t r = (send(socket, buffer, rest, 0));
          rounds++;
          if (r < 0) {
            perror("SEND FAILED.");
            return false;
          }
        } else {
          endmem += CHUNK_SIZE;
          copy(memblock + beginmem, memblock + endmem, buffer);
          beginmem = endmem;
          ssize_t r = (send(socket, buffer, 10000, 0));
          rounds++;
          if (r < 0) {
            perror("SEND FAILED.");
            return false;
          }
        }
      }
      return true;

    } else {
      perror("SEND SIZE ERROR");
      return false;
    }
  }
}

string KClientT2V1::receiveMessage(const int &socket, int buffersize) {
  char buffer[buffersize];
  string message;
  if (recv(socket, buffer, static_cast<size_t>(buffersize), 0) < 0) {
    perror("RECEIVE FAILED");
  }
  message = buffer;
  message.erase(static_cast<unsigned long>(buffersize));
  this->log(socket, "---> " + message);
  return message;
}

ifstream KClientT2V1::receiveStream(int socketFD, string filename) {
  uint32_t size;
  auto *data = (char *)&size;
  if (recv(socketFD, data, sizeof(uint32_t), 0) < 0) {
    perror("RECEIVE SIZE ERROR");
  }

  ntohl(size);
  this->log(socketFD, "--> SIZE: " + to_string(size));
  this->sendMessage("SIZE-OK", socketFD);

  auto *memblock = new char[size];
  ssize_t expected_data = size;
  ssize_t received_data = 0;
  while (received_data < expected_data) {
    ssize_t data_fd = recv(socketFD, memblock + received_data, 10000, 0);
    received_data += data_fd;
  }
  print(received_data);

  if (received_data != expected_data) {
    perror("RECEIVE STREAM ERROR");
    exit(1);
  }

  ofstream temp(filename, ios::out | ios::binary);
  temp.write(memblock, size);
  temp.close();
  return ifstream(filename);
}

void KClientT2V1::log(int socket, string message) {
  if (this->verbose) {
    sockaddr address;
    socklen_t addressLength;
    sockaddr_in *addressInternet;
    string ip;
    int port;
    getpeername(socket, &address, &addressLength);
    addressInternet = (struct sockaddr_in *)&address;
    ip = inet_ntoa(addressInternet->sin_addr);
    port = addressInternet->sin_port;
    string msg = "[" + ip + ":" + to_string(port) + "] " + message;
    print(msg);
  }
}

ifstream KClientT2V1::pkCToStream() {
  ofstream filedat("pk.dat");
  Export(filedat, this->fhesiPubKey->GetRepresentation());
  return ifstream("pk.dat", ios::binary);
}

ifstream KClientT2V1::ksCToStream() {
  ofstream filedat("ksC.dat");
  Export(filedat, this->keySwitchSI->GetRepresentation());
  return ifstream("ksC.dat");
}

ifstream KClientT2V1::ksTToStream() {
  ofstream filedat("ksT.dat");
  Export(filedat, this->keySwitchSIT->GetRepresentation());
  return ifstream("ksT.dat");
}

ifstream KClientT2V1::skTToStream() {
  ofstream filedat("skT.dat");
  Export(filedat, this->fhesiSecKeyT->GetRepresentation());
  return ifstream("skT.dat");
}

ifstream KClientT2V1::contextToStream() {
  ofstream filedat("context.dat");
  this->client_context->ExportSIContext(filedat);
  return ifstream("context.dat");
}

ifstream KClientT2V1::encryptedDataToStream(const Ciphertext &ciphertext) {
  ofstream ofstream1("temp.dat");
  Export(ofstream1, ciphertext);
  return ifstream("temp.dat");
}

void KClientT2V1::sendEncryptionParamToTServer() {
  this->sendMessage("C-PK", this->t_serverSocket);
  string message = this->receiveMessage(this->t_serverSocket, 10);
  if (message != "T-PK-READY") {
    perror("ERROR IN PROTOCOL 2-STEP 1");
    return;
  }
  this->sendStream(this->pkCToStream(), this->t_serverSocket);
  string message1 = this->receiveMessage(this->t_serverSocket, 13);
  if (message1 != "T-PK-RECEIVED") {
    perror("ERROR IN PROTOCOL 2-STEP 2");
    return;
  }
  this->sendMessage("C-SMT", this->t_serverSocket);
  string message2 = this->receiveMessage(this->t_serverSocket, 11);
  if (message2 != "T-SMT-READY") {
    perror("ERROR IN PROTOCOL 2-STEP 3");
    return;
  }
  this->sendStream(this->ksTToStream(), this->t_serverSocket);
  string message3 = this->receiveMessage(this->t_serverSocket, 14);
  if (message3 != "T-SMT-RECEIVED") {
    perror("ERROR IN PROTOCOL 2-STEP 4");
    return;
  }
  this->sendMessage("C-SKT", this->t_serverSocket);
  string message4 = this->receiveMessage(this->t_serverSocket, 11);
  if (message4 != "T-SKT-READY") {
    perror("ERROR IN PROTOCOL 2-STEP 5");
    return;
  }
  this->sendStream(this->skTToStream(), this->t_serverSocket);
  string message5 = this->receiveMessage(this->t_serverSocket, 14);
  if (message5 != "T-SKT-RECEIVED") {
    perror("ERROR IN PROTOCOL 2-STEP 6");
    return;
  }
  this->sendMessage("C-CONTEXT", this->t_serverSocket);
  string message6 = this->receiveMessage(this->t_serverSocket, 9);
  if (message6 != "T-C-READY") {
    perror("ERROR IN PROTOCOL 2-STEP 7");
    return;
  }
  this->sendStream(this->contextToStream(), this->t_serverSocket);
  string message7 = this->receiveMessage(this->t_serverSocket, 12);
  if (message7 != "T-C-RECEIVED") {
    perror("ERROR IN PROTOCOL 2-STEP 8");
    return;
  }
  print("PROTOCOL 2 COMPLETED");
  close(this->t_serverSocket);
}

void KClientT2V1::sendEncryptionParamToUServer() {
  this->sendMessage("C-PK", this->u_serverSocket);
  string message = this->receiveMessage(this->u_serverSocket, 10);
  if (message != "U-PK-READY") {
    perror("ERROR IN PROTOCOL 1-STEP 1");
    return;
  }
  this->sendStream(this->pkCToStream(), this->u_serverSocket);
  string message1 = this->receiveMessage(this->u_serverSocket, 13);
  if (message1 != "U-PK-RECEIVED") {
    perror("ERROR IN PROTOCOL 1-STEP 2");
    return;
  }
  this->sendMessage("C-SM", this->u_serverSocket);
  string message2 = this->receiveMessage(this->u_serverSocket, 10);
  if (message2 != "U-SM-READY") {
    perror("ERROR IN PROTOCOL 1-STEP 3");
    return;
  }
  this->sendStream(this->ksCToStream(), this->u_serverSocket);
  string message3 = this->receiveMessage(this->u_serverSocket, 13);
  if (message3 != "U-SM-RECEIVED") {
    perror("ERROR IN PROTOCOL 1-STEP 4");
    return;
  }
  this->sendMessage("C-CONTEXT", this->u_serverSocket);
  string message4 = this->receiveMessage(this->u_serverSocket, 9);
  if (message4 != "U-C-READY") {
    perror("ERROR IN PROTOCOL 1-STEP 5");
    return;
  }
  this->sendStream(this->contextToStream(), this->u_serverSocket);
  string message5 = this->receiveMessage(this->u_serverSocket, 12);
  if (message5 != "U-C-RECEIVED") {
    perror("ERROR IN PROTOCOL 2-STEP 8");
    return;
  }
  print("PROTOCOL 1 COMPLETED");
  close(this->u_serverSocket);
  this->u_serverSocket = -1;
}

void KClientT2V1::sendEncryptedDataToUServer() {
  this->connectToUServer();
  this->sendMessage("C-DA", this->u_serverSocket);
  string message = this->receiveMessage(this->u_serverSocket, 12);
  if (message != "U-DATA-READY") {
    perror("ERROR IN PROTOCOL 3-STEP 1");
    return;
  }
  auto numberofpoints = static_cast<uint32_t>(this->loadeddata.size());
  htonl(numberofpoints);
  if (0 > send(this->u_serverSocket, &numberofpoints, sizeof(uint32_t), 0)) {
    perror("ERROR IN PROTOCOL 3-STEP 1.2");
    return;
  }
  string message12 = this->receiveMessage(this->u_serverSocket, 12);
  if (message12 != "U-N-RECEIVED") {
    perror("ERROR IN PROTOCOL 3-STEP 1.3");
    return;
  }
  for (auto &iter : this->encrypted_data_hash_table) {
    log(this->u_serverSocket, "<--- POINT-" + to_string(iter.first));
    this->sendMessage("C-DATA-P", this->u_serverSocket);
    string message1 = this->receiveMessage(this->u_serverSocket, 14);
    if (message1 != "U-DATA-P-READY") {
      perror("ERROR IN PROTOCOL 3-STEP 2");
      return;
    }

    uint32_t pointid = iter.first;
    htonl(pointid);
    if (0 > send(this->u_serverSocket, &pointid, sizeof(uint32_t), 0)) {
      perror("ERROR IN PROTOCOL 3-STEP 3");
      return;
    }
    string message2 = this->receiveMessage(this->u_serverSocket, 14);
    if (message2 != "U-P-I-RECEIVED") {
      perror("ERROR IN PROTOCOL 3-STEP 4");
      return;
    }

    Ciphertext ciphertext(*this->fhesiPubKey);
    Plaintext plaintext(*this->client_context, iter.second);
    this->fhesiPubKey->Encrypt(ciphertext, plaintext);
    this->sendStream(this->encryptedDataToStream(ciphertext),
                     this->u_serverSocket);

    string message3 = this->receiveMessage(this->u_serverSocket, 17);
    if (message3 != "U-DATA-P-RECEIVED") {
      perror("ERROR IN PROTOCOL 3-STEP 3");
      return;
    }
  }
  this->sendMessage("C-DATA-E", this->u_serverSocket);
  string message1 = this->receiveMessage(this->u_serverSocket, 15);
  if (message1 != "U-DATA-RECEIVED") {
    perror("ERROR IN PROTOCOL 3-STEP 4");
    return;
  }
  print("PROTOCOL 3 COMPLETED");
}

void KClientT2V1::receiveResult() {
  print("WAITING FOR KMEANS RESULTS");

  this->sendMessage("C-READY", this->u_serverSocket);
  uint32_t k_factor;
  auto *data = (char *)&k_factor;
  if (recv(this->u_serverSocket, data, sizeof(uint32_t), 0) < 0) {
    perror("RECEIVE K ERROR. ERROR IN PROTOCOL 8.3-STEP 2");
  }
  ntohl(k_factor);
  this->log(this->u_serverSocket, "--> K-MEANS K: " + to_string(k_factor));
  for (unsigned i = 0; i < this->encrypted_data_hash_table.size(); i++) {
    string message1 = this->receiveMessage(this->u_serverSocket, 3);
    if (message1 != "U-P") {
      perror("ERROR IN PROTOCOL 8.3-STEP 3");
      return;
    }
    this->sendMessage("C-P-R", this->u_serverSocket);
    uint32_t identifier;
    auto *data1 = (char *)&identifier;
    if (recv(this->u_serverSocket, data1, sizeof(uint32_t), 0) < 0) {
      perror("RECEIVE IDENTITY ERROR. ERROR IN PROTOCOL 8.3-STEP 4");
    }
    ntohl(identifier);
    this->log(this->u_serverSocket, "--> POINT ID: " + to_string(identifier));
    this->sendMessage("P-I-R", this->u_serverSocket);
    vector<long> point_results;
    for (unsigned j = 0; j < k_factor; j++) {
      string filename = "cluster_" + to_string(j) + ".dat";
      ifstream cipher = this->receiveStream(this->u_serverSocket, filename);
      Ciphertext ciphertext(*this->fhesiPubKey);
      ifstream in(filename);
      Import(in, ciphertext);
      Plaintext plaintext;
      this->fhesiSecKey->Decrypt(plaintext, ciphertext);
      ZZ_pX index_of_cluster = plaintext.message;
      ZZ_p ind_of_cluster;
      ind_of_cluster = coeff(index_of_cluster, 0);
      const ZZ &x = rep(ind_of_cluster);
      long t;
      print(t);
      t = to_long(x);
      point_results.push_back(t);
      this->sendMessage("P-CI-R", this->u_serverSocket);
    }
    this->results[identifier] = point_results;

    string message2 = this->receiveMessage(this->u_serverSocket, 7);
    if (message2 != "U-R-P-E") {
      perror("ERROR IN PROTOCOL 8.3-STEP 5");
      return;
    }
  }
  string message2 = this->receiveMessage(this->u_serverSocket, 10);
  if (message2 != "U-RESULT-E") {
    perror("ERROR IN PROTOCOL 8.1-STEP 5");
    return;
  }
  this->sendMessage("C-END", this->u_serverSocket);
  close(this->u_serverSocket);
  this->u_serverSocket = -1;
  print("--------------------RESULTS--------------------");
  for (auto &iter : this->encrypted_data_hash_table) {
    unsigned result = 4000;
    for (unsigned l = 0; l < k_factor; l++) {
      if (this->results[iter.first][l] != 0) {
        result = l;
      }
    }
    // cout << "Point ID: " << iter.first << " Cluster: " << result << endl;
    cout << result << endl;
  }
}

void KClientT2V1::createStruct() {
  srand(static_cast<unsigned int>(time(NULL)));
  for (unsigned i = 0; i < this->loadeddata.size(); i++) {
    ZZ_pX point = this->loadeddata[i];
    uint32_t identifier;
    identifier = static_cast<uint32_t>(rand());
    this->encrypted_data_hash_table[identifier] = point;
    this->identifiers[identifier] = identifier;
    // this->unencrypted_data_hash_table[identifier] = pointToInt;
  }
}

void KClientT2V1::calculateCentroid(int socketFD) {
  this->sendMessage("C-NC-READY", socketFD);
  for (unsigned i = 0; i < this->k; i++) {
    uint32_t cluster_index;
    auto *data = (char *)&cluster_index;
    if (recv(socketFD, data, sizeof(uint32_t), 0) < 0) {
      perror("RECEIVE CLUSTER INDEX ERROR");
    }
    ntohl(cluster_index);
    this->sendMessage("C-RECEIVED-CI", socketFD);
    Ciphertext cluster_size_encrypted(*this->fhesiPubKey);
    this->receiveStream(socketFD, to_string(i) + "_clustersize.dat");
    ifstream in(to_string(i) + "_clustersize.dat");
    Import(in, cluster_size_encrypted);
    Plaintext cluster_size_decrypted;
    this->fhesiSecKey->Decrypt(cluster_size_decrypted, cluster_size_encrypted);
    long cluster_size = this->extractClusterSize(cluster_size_decrypted);
    print("The cluster size is: " + to_string(cluster_size));
    if (cluster_size == 0) {
      cluster_size = 1;
    }
    this->sendMessage("C-RECEIVED-CS", socketFD);
    Ciphertext centroid_sum(*this->fhesiPubKey);
    this->receiveStream(socketFD, to_string(cluster_index) + "centroidsum.dat");
    ifstream in1(to_string(cluster_index) + "centroidsum.dat");
    Import(in1, centroid_sum);
    this->sendMessage("C-RECEIVED-C", socketFD);

    Plaintext pcentroidsum;
    this->fhesiSecKey->Decrypt(pcentroidsum, centroid_sum);
    Plaintext newcentroid = this->newCentroid(pcentroidsum, cluster_size);
    Ciphertext cnewcnetroid(*this->fhesiPubKey);
    this->fhesiPubKey->Encrypt(cnewcnetroid, newcentroid);
    this->sendStream(this->centroidCoefToStream(cnewcnetroid), socketFD);

    string message2 = this->receiveMessage(socketFD, 13);
    if (message2 != "U-NC-RECEIVED") {
      perror("ERROR IN PROTOCOL 6-STEP 4");
      return;
    }
  }
  string message3 = this->receiveMessage(socketFD, 11);
  if (message3 != "U-C-UPDATED") {
    perror("ERROR IN PROTOCOL 6-STEP 5");
    return;
  }

  this->sendMessage("C-READY", socketFD);
  print("K-MEANS ROUND FINISH");
}

Plaintext KClientT2V1::newCentroid(const Plaintext &sum, long mean) {
  ZZ_pX centroidx = sum.message;

  ZZ_pX new_centroid = centroidx / mean;

  Plaintext centroid(*this->client_context, new_centroid);
  return centroid;
}

ifstream KClientT2V1::centroidCoefToStream(const Ciphertext &centroid) {
  ofstream ofstream1("centroidcoef.dat");
  Export(ofstream1, centroid);
  return ifstream("centroidcoef.dat");
}

long KClientT2V1::extractClusterSize(const Plaintext &clustersize) {
  ZZ_pX clustersize_1 = clustersize.message;
  ZZ_p coef;
  coef = coeff(clustersize_1, 0);
  const ZZ &x = rep(coef);
  return to_long(x);
}