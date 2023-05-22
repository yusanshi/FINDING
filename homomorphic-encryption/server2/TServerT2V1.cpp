//
// Created by george on 16/11/2017.
//

#include "TServerT2V1.h"

TServerT2V1::TServerT2V1(string t_serverIP, int t_serverPort, bool verbose) {
  this->active = true;
  this->t_serverIP = move(t_serverIP);
  this->t_serverPort = t_serverPort;
  this->verbose = verbose;
  print("TRUSTED SERVER");
  this->socketCreate();
  this->socketBind();

  this->socketListen();
  this->socketAccept();
  print("CLIENT ENCRYPTION PARAMETERS");
  ifstream contextfile("context.dat");
  FHEcontext fhEcontext(contextfile);
  this->client_context = &fhEcontext;
  activeContext = &fhEcontext;
  ifstream pkC("pkC.dat");
  FHESIPubKey fhesiPubKey(fhEcontext);
  fhesiPubKey.Import(pkC);
  this->client_pubkey = &fhesiPubKey;
  ifstream skT("skT.dat");
  FHESISecKey fhesiSecKey(fhEcontext);
  fhesiSecKey.Import(skT);
  this->t_server_seckey = &fhesiSecKey;
  ifstream smT("smT.dat");
  KeySwitchSI keySwitchSI(fhEcontext);
  keySwitchSI.Import(smT);
  this->t_server_SM = &keySwitchSI;
  print("CONTEXT");
  print(fhEcontext);
  print("CLIENT PUBLIC KEY");
  print(fhesiPubKey);
  print("TServerT2V1 SECRET KEY");
  print(fhesiSecKey);
  print("TServerT2V1 SWITCH MATRIX ");
  print(keySwitchSI);

  while (this->active) {
    this->socketAccept();
  }
}

void TServerT2V1::socketCreate() {
  this->t_serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (this->t_serverSocket < 0) {
    perror("ERROR IN SOCKET CREATION");
    exit(1);
  } else {
    int opt = 1;
    setsockopt(this->t_serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt,
               sizeof(opt));
    string message = "Socket created successfully. File descriptor: " +
                     to_string(this->t_serverSocket);
    print(message);
  }
}

void TServerT2V1::socketBind() {
  struct sockaddr_in u_serverAddress;
  u_serverAddress.sin_family = AF_INET;
  u_serverAddress.sin_port = htons(static_cast<uint16_t>(this->t_serverPort));
  u_serverAddress.sin_addr.s_addr = inet_addr(this->t_serverIP.data());
  if (bind(this->t_serverSocket, (sockaddr *)&u_serverAddress,
           sizeof(u_serverAddress)) < 0) {
    perror("BIND ERROR");
    exit(1);
  } else {
    string message = "Socket bound successfully to :[" + t_serverIP + ":" +
                     to_string(this->t_serverPort) + "]";
    print(message);
  }
}

void TServerT2V1::socketListen() {
  listen(this->t_serverSocket, 5);
  print("Server is listening...");
}

void TServerT2V1::socketAccept() {
  int socketFD;
  socketFD = accept(this->t_serverSocket, NULL, NULL);
  if (socketFD < 0) {
    perror("SOCKET ACCEPT ERROR");
  } else {
    print("CLIENT_CONNECTED");
    this->handleRequest(socketFD);
  }
}

void TServerT2V1::handleRequest(int socketFD) {
  string message = this->receiveMessage(socketFD, 4);
  if (message == "C-PK") {
    this->receiveEncryptionParamFromClient(socketFD);
  } else if (message == "U-KM") {
    this->initializeKM(socketFD);
  } else if (message == "U-DP") {
    this->classifyToCluster(socketFD);
  } else if (message == "UEKM") {
    this->sendMessage(socketFD, "T-END");
    this->active = false;
    print("TServerT2V1 STOP AND EXIT");
  } else {
    perror("ERROR IN PROTOCOL INITIALIZATION");
    return;
  }
}

bool TServerT2V1::sendStream(ifstream data, int socket) {
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

bool TServerT2V1::sendMessage(int socketFD, string message) {
  if (send(socketFD, message.c_str(), strlen(message.c_str()), 0) < 0) {
    perror("SEND FAILED.");
    return false;
  } else {
    this->log(socketFD, "<--- " + message);
    return true;
  }
}

string TServerT2V1::receiveMessage(int socketFD, int buffersize) {
  char buffer[buffersize];
  string message;
  if (recv(socketFD, buffer, static_cast<size_t>(buffersize), 0) < 0) {
    perror("RECEIVE FAILED");
  }
  message = buffer;
  message.erase(static_cast<unsigned long>(buffersize));
  this->log(socketFD, "---> " + message);
  return message;
}

ifstream TServerT2V1::receiveStream(int socketFD, string filename) {
  uint32_t size;
  auto *data = (char *)&size;
  if (recv(socketFD, data, sizeof(uint32_t), 0) < 0) {
    perror("RECEIVE SIZE ERROR");
  }

  ntohl(size);
  this->log(socketFD, "--> SIZE: " + to_string(size));
  this->sendMessage(socketFD, "SIZE-OK");

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

void TServerT2V1::log(int socket, string message) {
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

void TServerT2V1::receiveEncryptionParamFromClient(int socketFD) {
  this->sendMessage(socketFD, "T-PK-READY");
  this->receiveStream(socketFD, "pkC.dat");
  this->sendMessage(socketFD, "T-PK-RECEIVED");
  string message = this->receiveMessage(socketFD, 5);
  if (message != "C-SMT") {
    perror("ERROR IN PROTOCOL 2-STEP 2");
    return;
  }
  this->sendMessage(socketFD, "T-SMT-READY");
  this->receiveStream(socketFD, "smT.dat");
  this->sendMessage(socketFD, "T-SMT-RECEIVED");
  string message1 = this->receiveMessage(socketFD, 5);
  if (message1 != "C-SKT") {
    perror("ERROR IN PROTOCOL 2-STEP 3");
    return;
  }
  this->sendMessage(socketFD, "T-SKT-READY");
  this->receiveStream(socketFD, "skT.dat");
  this->sendMessage(socketFD, "T-SKT-RECEIVED");
  string message2 = this->receiveMessage(socketFD, 9);
  if (message2 != "C-CONTEXT") {
    perror("ERROR IN PROTOCOL 2-STEP 4");
    return;
  }
  this->sendMessage(socketFD, "T-C-READY");
  this->receiveStream(socketFD, "context.dat");
  this->sendMessage(socketFD, "T-C-RECEIVED");
  print("PROTOCOL 2 COMPLETED");
}

void TServerT2V1::initializeKM(int socketFD) {
  this->sendMessage(socketFD, "T-READY");
  uint32_t size;
  auto *data = (char *)&size;
  if (recv(socketFD, data, sizeof(uint32_t), 0) < 0) {
    perror("RECEIVE K ERROR");
  }
  ntohl(size);
  this->log(socketFD, "--> K: " + to_string(size));
  this->k = size;
  this->sendMessage(socketFD, "T-K-RECEIVED");
  uint32_t dimension;
  auto *data1 = (char *)&dimension;
  if (recv(socketFD, data1, sizeof(uint32_t), 0) < 0) {
    perror("RECEIVE DIMENSION ERROR");
  }
  ntohl(dimension);
  this->log(socketFD, "--> DIMENSION: " + to_string(dimension));
  this->dim = dimension;
  this->sendMessage(socketFD, "T-DIM-RECEIVED");
  print("PROTOCOL 4 COMPLETED");
}

void TServerT2V1::classifyToCluster(int socketFD) {
  this->sendMessage(socketFD, "T-READY");
  for (int i = 0; i < this->k; i++) {
    uint32_t index;
    auto *data = (char *)&index;
    if (recv(socketFD, data, sizeof(uint32_t), 0) < 0) {
      perror("RECEIVE INDEX ERROR");
    }
    ntohl(index);
    this->sendMessage(socketFD, "T-RECEIVED-CI");
    Ciphertext distance(*this->client_pubkey);
    this->receiveStream(socketFD, to_string(index) + ".dat");
    ifstream in(to_string(index) + ".dat");
    Import(in, distance);
    this->point_distances[index] = distance;
    this->sendMessage(socketFD, "T-D-RECEIVED");
  }
  string message = this->receiveMessage(socketFD, 5);
  if (message != "U-R-I") {
    perror("ERROR IN PROTOCOL 5-STEP 1");
    return;
  }
  unsigned index = extractClusterIndex();
  ZZ_pX zeroindexPX;
  SetCoeff(zeroindexPX, 0, 0);
  Plaintext plain_zero(*this->client_context, zeroindexPX);
  Ciphertext cipher_zero(*this->client_pubkey);
  this->client_pubkey->Encrypt(cipher_zero, plain_zero);
  ZZ_pX unitindexPX;
  SetCoeff(unitindexPX, 0, 1);
  Plaintext plain_unit(*this->client_context, unitindexPX);
  Ciphertext cipher_unit(*this->client_pubkey);
  this->client_pubkey->Encrypt(cipher_unit, plain_unit);
  for (unsigned j = 0; j < this->k; j++) {
    if (index == j) {
      this->sendStream(this->indexToStream(cipher_unit), socketFD);
    } else {
      this->sendStream(this->indexToStream(cipher_zero), socketFD);
    }
    string message1 = this->receiveMessage(socketFD, 7);
    if (message1 != "U-R-E-I") {
      perror("ERROR IN PROTOCOL 5-STEP 2");
      return;
    }
  }
  string message2 = this->receiveMessage(socketFD, 12);
  if (message2 != "U-RECEIVED-I") {
    perror("ERROR IN PROTOCOL 5-STEP 3");
    return;
  }
}

unsigned TServerT2V1::extractClusterIndex() {
  map<unsigned, long> distancesManhattan;
  ZZ p = this->client_context->ModulusP();

  for (unsigned i = 0; i < this->k; i++) {
    Plaintext pdistance;
    Ciphertext cdistance = this->point_distances[i];
    this->t_server_SM->ApplyKeySwitch(cdistance);
    this->t_server_seckey->Decrypt(pdistance, cdistance);
    distancesManhattan[i] = extractHM(pdistance, p);
  }
  unsigned index = 0;
  long min = distancesManhattan[index];
  for (unsigned i = 0; i < this->k; i++) {
    if (min > distancesManhattan[i]) {
      index = i;
      min = distancesManhattan[i];
    }
  }
  return index;
}

ifstream TServerT2V1::indexToStream(const Ciphertext &index) {
  ofstream ofstream1("index.dat");
  Export(ofstream1, index);
  return ifstream("index.dat");
}