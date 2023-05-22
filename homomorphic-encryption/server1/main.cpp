
#include <chrono>
#include <ctime>
#include <iomanip>

#include "UServerT2V1.h"
int main() {
  UServerT2V1 server("127.0.0.1", 5001, "127.0.0.1", 5002, 3);
  return 0;
}
