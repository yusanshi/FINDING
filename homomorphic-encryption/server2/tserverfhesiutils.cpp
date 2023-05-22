//
// Created by George Sakellariou on 29/09/2017.
//

#include "tserverfhesiutils.h"

using namespace std;

long extractHM(const Plaintext &distance, ZZ &p) {
    ZZ_pX dp =distance.message;
    ZZ_p dhm;
    ZZ hmd=to_ZZ(0);
    for(long i=0;i<dp.rep.length();i++){
        dhm=coeff(dp,i);
        ZZ x= rep(dhm);
        if(x>p/2){
            ZZ t= x-p;
            t*=-1;
            hmd+=t;
        }else{
            hmd+=x;
        }
    }
    return to_long(hmd);
}

long extractHM1(const vector<Plaintext> &distance, ZZ &p){
    long dimension = distance.size();
    long result=0;
    for(unsigned i=0;i<dimension;i++){
        result+=extractHM(distance[i],p);
    }
    return result;

}

long extractDistance(const Plaintext &distance) {
    ZZ_pX dp =distance.message;
    ZZ_p dhm;
    ZZ hmd;
    dhm=coeff(dp,0);
    hmd= rep(dhm);
    return to_long(hmd);
}

void timeCalulator(const clock_t &c_start, const chrono::high_resolution_clock::time_point &t_start) {
    std::clock_t c_end = std::clock();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << fixed << setprecision(2) << "CPU time used: "
              << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << " ms\n"
              << "Wall clock time passed: "
              << chrono::duration<double, milli>(t_end-t_start).count()
              << " ms"<<endl;
}

