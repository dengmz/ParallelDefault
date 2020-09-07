#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <chrono>

/* Matrix size */
#define Nb 100
#define Ny 300
#define lbd -1
#define ubd 0
#define tol 1e-6
#define rstar 0.017
#define alpha 0.5
#define maxInd Ny*Nb
#define beta 0.953
#define phi 0.282
#define tau 0.5
#define delta 0.8
#define rho 0.9 
#define sigma 0.025

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
void tauchen(double *restrict P){
    double sigma_z = sqrt(sigma*sigma/(1.0-rho*rho));
    double step = 10*sigma_z/double(Ny-1);
    std::vector<double> Z(Ny, 0.0);
    Z[Ny-1] = 5.0*sigma_z; Z[0] = -5.0*sigma_z;
    for (int i = 2; i <= Ny-1; i++){
        Z[i-1] = Z[i-2] + step;
    }
    //fill front and back
    for(int i=0; i < Ny; i++){
        P[i] = mynormcdf((Z[0]-rho*Z[i]+step/2)/sigma);
	    P[i+Ny*(Ny-1)] = 1.0 - mynormcdf( (Z[Ny-1]-rho*Z[i]-step/2)/sigma);
    }

    //fill the middle
    for (int i = 0; i < Ny; ++i) {
	    for (int iz = 1; iz < Ny-1; ++iz) {
            P[i+Ny*iz] = mynormcdf( (Z[iz]-rho*Z[i]+step/2.0)/sigma)-mynormcdf( (Z[iz]-rho*Z[i]-step/2)/sigma  );
	    }
    }
};




int main(){
    
    auto start = std::chrono::high_resolution_clock::now();

    int size = 0;

    double *restrict B;
    size = Nb*sizeof(double);
    B = (double*) malloc(size);
    double minB = lbd ; // minp*kss;
	double maxB = ubd ;  // maxp*kss;
	double step = (maxB-minB)/double(Nb-1);
	for (int i_b = 0; i_b < Nb; i_b++) {
		B[i_b] = minB + step*double(i_b);
	};

    //for (int i = 0; i < Nb; i++) {std::cout<<B[i] <<' ';}
    //std::cout<< std::endl; 
    // return 0;

    double *restrict Y;
    size = Ny*sizeof(double);
    Y = (double*) malloc(size);
    double sigma_z = sqrt(sigma*sigma/(1.0-rho*rho));
    double step2 = 10*sigma_z/(Ny-1);
    for (int i_y = 0; i_y < Ny; i_y++) {
		Y[i_y] = -5*sigma_z + step2*double(i_y);
	};

    double *restrict P = (double*) malloc(Ny*Ny*sizeof(double));
    //for (int i = 0; i < Ny*Ny*sizeof(double); i++){P[i] = 0.0;};
    tauchen(P);
    /*
    for (int i = 0; i < Ny*Ny; i++){
        if((i+1)%Ny==1) std::cout<<std::endl;
        std::cout<<P[i]<<' ';}
    
    */
    double *restrict V = (double*) malloc(Ny*Nb*sizeof(double));
    for (int i = 0; i < Ny*Nb; i++){V[i] = (1.0/((1.0-beta)*(1.0-alpha)));};

    double *restrict Price = (double*) malloc(Ny*Nb*sizeof(double));
    for (int i = 0; i < Ny*Nb; i++){Price[i] = (1.0/(1.0+rstar));};

    double *restrict Vr = (double*) malloc(Ny*Nb*sizeof(double));
    for (int i = 0; i < Ny*Nb; i++){Vr[i] = 0.0;};

    double *restrict Vd = (double*) malloc(Ny*sizeof(double));
    for (int i = 0; i < Ny; i++){Vd[i] = 0.0;};

    double *restrict decision = (double*) malloc(Ny*Nb*sizeof(double));
    for (int i = 0; i < Ny*Nb; i++){decision[i] = 1.0;};

    double *restrict prob = (double*) malloc(Ny*Nb*sizeof(double));
    for (int i = 0; i < Ny*Nb; i++){prob[i] = 0.0;};

    double err = 100.0;
    int iter = 1;

    while (iter<300){
    
        double *restrict V0 = (double*) malloc(Ny*Nb*sizeof(double));
        for (int i = 0; i < Ny*Nb; i++) {V0[i] = V[i];}

        double *restrict Vd0 = (double*) malloc(Ny*sizeof(double));
        for (int i = 0; i < Ny; i++){Vd0[i] = Vd[i];};

        double *restrict Price0 = (double*) malloc(Ny*Nb*sizeof(double));
        for (int i = 0; i < Ny*Nb; i++){Price0[i] = Price[i];};


        //default value
        //std::cout<<"defualt begins:"<<std::endl;
        #pragma acc parallel loop
        for (int iy = 0; iy < Ny; iy++){        
            double sumdef = 0.0;
            double consdef = pow(exp(Y[iy]*(1.0-tau)), (1.0-alpha))/(1.0-alpha);
            for (int y = 0; y < Ny; y++){
                sumdef += beta* P[iy + Ny*y] * (phi* V0[y] + (1.0-phi)*Vd0[y]);
            }
            Vd[iy] = sumdef + consdef;
        }
        
        //for (int i = 0; i < Ny; i++){std::cout<<Y[i]<<' ';};

        //repayment value
        //std::cout<<"repay begins:"<<std::endl;
        #pragma acc parallel loop collapse(2)
        for (int ib = 0; ib < Nb; ib++){
            for (int iy = 0; iy < Ny; iy++){
                double Max = -100000.0;
                for (int b = 0; b < Nb; b++){
                    
                    double c = exp(Y[iy]) + B[ib] - Price0[iy+b*Ny]* B[b];
                    if (c > 0){
                        double sumret = 0.0;
                        for (int y = 0; y < Ny; y++){
                            sumret += V0[y+b*Ny] * P[iy+y*Ny];
                        };
                        double vr = pow(c, (1.0-alpha))/(1.0-alpha) + beta* sumret;
                        if (Max < vr) Max = vr;
                    }
                    Vr[iy+ib*Ny] = Max;
                }
            }
        }
        /*
        for (int i = 0; i < Ny*Nb; i++){
            if((i+1)%Ny==1) std::cout<<std::endl;
            std::cout<<Vr[i]<<' ';}
        */

        //decide
        //std::cout<<"decide begins:"<<std::endl;
        for (int i = 0; i < Ny*Nb; i++){prob[i] = 0.0;}; //set prob to 0
        #pragma acc parallel loop collapse(2)
        for (int ib = 0; ib < Nb; ib++){
            for (int iy = 0; iy < Ny; iy++){

            if (Vd[iy] < Vr[iy+ib*Ny]){
                V[iy+ib*Ny] = Vr[iy+ib*Ny];
                decision[iy+ib*Ny] = 0;
            }
            else {
                V[iy+ib*Ny] = Vd[iy];
                decision[iy+ib*Ny] = 1;
            }

            for (int y = 0; y < Ny; y++){
                prob[iy+ib*Ny] += P[iy+y*Ny] * decision[y+ib*Ny];
            }

            Price[iy+ib*Ny] = (1.0-prob[iy+ib*Ny]) / (1.0+rstar);

            }
        }

        //price_error
        //std::cout<<"diff begins:"<<std::endl;
        double diff = 0.0;
        for (int i=0; i < Ny*Nb; i++){
            diff = std::max(abs(Price[i]-Price0[i]),diff);
        }

        err = diff;
        

        //saxpy
        //std::cout<<"saxpy begins:"<<std::endl;
        #pragma acc parallel loop
        for (int i = 0; i < Ny*Nb; i++){
            V[i] = (1.0-delta)*V0[i] + delta*V[i];
        }
        
        #pragma acc parallel loop
        for (int i = 0; i < Ny*Nb; i++){
            Price[i] = (1.0-delta)*Price0[i] + delta*Price[i];
        }

        #pragma acc parallel loop
        for (int i = 0; i < Ny; i++){
            Vd[i] = (1.0-delta)*Vd0[i] + delta*Vd[i];
        }

        std::cout << "=====iter=" << iter << ", err=" << err <<std::endl;
        iter++;
    }

    //recording time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
    std::cout<<"number of microseconds: " << duration.count()<<std::endl;

    // Save results
    std::ofstream value_out, price_out, decision_out;
    value_out.open("acc_my_value.csv"); // open file to print optimal
    price_out.open("acc_my_my_price.csv");
    decision_out.open("acc_my_decision.csv");


    for (int i = 1; i < (Ny*Nb+1); i++) {
        if(i%Ny == 0){
            value_out << '\n';
            price_out << '\n';
            decision_out << '\n';
        }
        else{
            value_out << V[i] << ',';
            price_out << Price[i] << ',';
            decision_out << decision[i] << ',';
        }
        
    }
    
    value_out.close();
    price_out.close();
    decision_out.close();
}