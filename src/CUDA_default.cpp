/*
 * File:   main.cpp
 * Author: Pablo A. Guerron
 *
 * Created on December 31, 2013, 10:06 AM
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <tuple>
#include <cstdlib>
#include <cmath>

/* Includes, system */
#include <iomanip>
#include <fstream>

#define nb 300
#define ny 21
#define lbd -1
#define ubd 0.0
#define rrisk 2.0
 
#define bbeta 0.953
#define ppi 0.15
#define ttheta 0.282

#define tol 1e-10

using namespace std ;
 
// A stand alone normcdf
double mynormcdf(double x) {
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);
    
    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    
    return 0.5*(1.0 + sign*y);
};

// Tauchen's method
void tauchen(double rrho, double ssigma, vector<double>& Z, vector<double>& P) {
    double ssigma_z = sqrt(pow(ssigma,2)/(1-pow(rrho,2)) );
    int nzgrid = Z.size();
    Z[nzgrid-1] = 5*ssigma_z; Z[0] = -5*ssigma_z;
    double step = (Z[nzgrid-1] - Z[0])/ double(nzgrid-1);
    for (int i = 2; i <= nzgrid-1; i++) {
        Z[i-1] = Z[i-2] + step;
    };
    
    for (int i_z = 0; i_z < nzgrid; ++i_z) {
        P[i_z] = mynormcdf( (Z[0]-rrho*Z[i_z]+step/2)/ssigma  );
        P[i_z+nzgrid*(nzgrid-1)] = 1.0 - mynormcdf( (Z[nzgrid-1]-rrho*Z[i_z]-step/2)/ssigma  );
    };
    
    for (int i_z = 0; i_z < nzgrid; ++i_z) {
        for (int i_zplus = 1; i_zplus < nzgrid-1; ++i_zplus) {
            P[i_z+nzgrid*i_zplus] = mynormcdf( (Z[i_zplus]-rrho*Z[i_z]+step/2)/ssigma  )-mynormcdf( (Z[i_zplus]-rrho*Z[i_z]-step/2)/ssigma  );
        };
    };
    
};

int fit2grid(const double x, const int n, const vector<double> &X) {
    if (x < X[0]) {
        return 0;
    } else if (x > X[n-1]) {
        return n-1;
    } else {
        int left=0; int right=n-1; int mid=(n-1)/2;
        while(right-left>1) {
            mid = (left + right)/2;
            if (X[mid]==x) {
                return mid;
            } else if (X[mid]<x) {
                left = mid;
            } else {
                right = mid;
            };

        };
        return left;
    }
};

double findmax(const int left_ind,const int right_ind, const int i_b,
        const int i_y, const vector<double> &B, const vector<double> &qrB, const vector<double> &Y, const vector<double> &EV) {
    double budget= B[i_b] + exp(Y[i_y]) ;
        
        if (right_ind==0){
            return (budget<qrB[right_ind]) ? double(-10000000) : pow(budget-qrB[right_ind],(1-rrisk))/(1-rrisk)+bbeta*EV[right_ind];
        }
        
        vector<double> auxi ;
        auxi.assign(right_ind,-10000000.0) ;
        double cons = 0.0 ;
        
        for (int bpx = 0 ; bpx < right_ind ; bpx++) {
            cons = budget - qrB[bpx] ;
            if (cons > 0) auxi[bpx] = pow(cons,(1-rrisk))/(1-rrisk)+bbeta*EV[bpx] ;
        }
        auto it = max_element(auxi.begin(),auxi.end()) ;
        return *it ;
        
}

// valuemax = findmaxnovect(left_ind, right_ind, i_b, i_y, B, qprB, Y, suma) ;9
double findmaxnovect(const int left_ind,const int right_ind, const int i_b,
        const int i_y, const vector<double> &B, const vector<double> &qrB, const vector<double> &Y, const vector<double> &EV) {
        
        double budget= B[i_b] + exp(Y[i_y]) ;
        
        double vlar = 0.0, vfirst = 0.0 ;
        
        int largest = left_ind, first = left_ind ;
        
        while (first < right_ind){
            
            (budget > qrB[largest]) ? vlar = pow(budget-qrB[largest],(1-rrisk))/(1-rrisk)+bbeta*EV[largest] : vlar = -10000000.0 ;
            (budget > qrB[first]) ? vfirst = pow(budget-qrB[first],(1-rrisk))/(1-rrisk)+bbeta*EV[first] : vfirst = -10000000.0 ;
            
            if (vlar < vfirst)
                largest = first ;
            first++ ;
        }
        return vlar ;
        
        
        
}

struct saxpy_functor
{
    const double a;

    saxpy_functor(double _a) : a(_a) {}

        double operator()(const double& x, const double& y) const {
            return (1-a) * x + a * y;
        }
};

void saxpy_fast(double A, vector<double>& X, vector<double>& Y)
{
    // Y <- A * X + Y
    transform(X.begin(), X.end(), Y.begin(),Y.begin(),saxpy_functor(A));
}


int main(int argc, char** argv) {

    const double alpha = 1.0;
    const double beta = 0.0;
    const double rstar = 0.017;
    const double lamupd = 0.5 ;
    const double lampudp = 1.0 - lamupd ;

    vector<double> B(nb); // bond grid
    vector<double> Y(ny); // endowment grid
    vector<double> P(ny*ny, 0.0);
    vector<double> qprice(ny*nb, 1.0 / (1 + rstar));
    vector<double> qprupd(ny*nb, 1.0 / (1 + rstar));
    vector<double> qtemp(ny*nb, 1.0 / (1 + rstar)), qbasura(ny*nb,0.0) ;
    vector<double> vbad(ny,0.0), vbadold(ny,0.0), value(ny*nb,1.0/((1 - bbeta)*(1 - rrisk)));
    vector<double> vtemp(ny*nb), valueold(ny*nb);
    vector<double> wndef(ny*nb), wdef(ny, 0.0), locoptndef(ny*nb, 0.0), decision(ny*nb, 1.0);
    vector<double> probdef(ny*nb, 1.0);


    // Initialize capital grid
    double minB = lbd; // minp*kss;
    double maxB = ubd; // maxp*kss;
    double step = (maxB - minB) / double(nb - 1);
    for (int i_b = 0; i_b < nb; i_b++) {
        B[i_b] = minB + step * double(i_b);
    };
    
    // initialize shock grid
    double rrho = 0.9 ;
    double ssigma = 0.025 ;
    tauchen(rrho, ssigma, Y, P); // discretizing a N(0,1) as epsilon
    double sumdef = 0.0 ;
    
    int index = nb*ny ;
    int left_ind = 0 ;
    double sumbad = 0.0 ;
    double consdef = 0.0 ;
    
    int iter = 0 ;
    vector<double> suma(nb,0.0) ;
    double err = 10, err1 = 10 ;
    double valuemax ;
    vector<double> qprB(nb,0.0) ;
    
    int i_b, i_y ;
    
    while (err > 1e-6 && iter < 1000){
        
    valueold = value ;
    vbadold  = vbad ;
  
    for (int iz = 0 ; iz < index ; iz++){
        i_b = iz / (ny);
        i_y = iz - i_b*ny;
        sumdef = 0.0 ;
        consdef = pow(exp(Y[i_y])*(1-ppi),(1-rrisk))/(1-rrisk) ;
        for (int ttx = 0; ttx < ny; ttx++) sumdef += P[i_y+ny*ttx]*((1-ttheta)*vbad[ttx]+ttheta*value[ttx+ny*(nb-1)]) ;
        wdef[i_y] = consdef + bbeta*sumdef ;
            
        qprB.assign(nb,0.0) ;
        // Create debt valuation vector
        for (int zx = 0; zx < nb; zx++){
            qprB[zx] = qprice[i_y + zx * ny] * B[zx] ;
        }

        int right_ind = nb ;
        
        suma.assign(nb,0.0) ;
        
        for (int lx = 0; lx < right_ind ; lx++){
            for (int ttx = 0; ttx < ny; ttx++) {
                    suma[lx] += P[i_y + ny * ttx] * value[ttx + lx * ny];
            }
        }
       
        valuemax = findmaxnovect(left_ind, right_ind, i_b, i_y, B, qprB, Y, suma) ;
        
        // Default?
        if (valuemax > wdef[i_y]){
            value[i_y+ny*i_b] = valuemax ;
            decision[i_y+ny*i_b] = 0 ;
        }
        else{
            value[i_y+ny*i_b] = wdef[i_y] ;
            decision[i_y+ny*i_b] = 1 ;
        }

        // Update prob of default
        sumdef = 0.0;
        for (int dfx = 0; dfx < ny; dfx++) sumdef += P[i_y+ny*dfx] * decision[dfx+ny*i_b];
        // Update debt price
        qprupd[i_y+ny*i_b] = (1.0 - sumdef) / (1 + rstar);
        
        sumbad = 0.0 ;
        for (int ttx = 0; ttx < ny; ttx++) sumbad += P[i_y+ny*ttx]*((1-ttheta)*vbad[ttx]+ttheta*value[ttx+ny*(nb-1)]) ;
        vbad[i_y] = consdef + sumbad*bbeta ;
        
    }
    
    transform(value.begin(),value.end(),valueold.begin(),vtemp.begin(),minus<double>()) ;

    for(auto it = vtemp.begin(); it != vtemp.end(); it++){
        *it = abs(*it) ;
    }
    err1 = *max_element(vtemp.begin(),vtemp.end()) ;
    
    // Diff price and update
    transform(qprupd.begin(),qprupd.end(),qprice.begin(),qtemp.begin(),minus<double>()) ;

    for(auto it = qtemp.begin(); it != qtemp.end(); it++){
        *it = abs(*it) ;
    }
    err = *max_element(qtemp.begin(),qtemp.end()) ;
    
    err = max(err,err1) ;
    
    cout << "Errors in iteration: " << iter << " is " << err << "and " << err1 << endl ;

    iter++ ;

    saxpy_fast(lampudp, qprupd, qprice) ;
    saxpy_fast(lamupd, valueold, value) ;
    saxpy_fast(lamupd, vbadold, vbad) ;
    
    };
    
    // Store solution

    // Save results
    ofstream policy_out, value_out, price_out, decision_out ;
    policy_out.open("qprice.csv");          // open file to print optimal
    value_out.open("vopt.csv");            // open file to print optimal
    price_out.open("moreprice.csv") ;
    decision_out.open("decision.csv") ;

    for (int i_e = 0; i_e < index; i_e++){
        policy_out << qprice[i_e] << '\n';
        value_out << value[i_e] << '\n';
        decision_out << decision[i_e] << '\n' ;
    }

    for(auto it = qprice.begin(); it != qprice.end(); it++){
        price_out << *it << '\n' ;
    }
    
    // finish saving
    policy_out.close() ;
    value_out.close() ;
    price_out.close() ;
    decision_out.close() ;
    
    return 0 ;
}
