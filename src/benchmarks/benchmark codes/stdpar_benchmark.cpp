#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <execution>
#include <numeric>
#include <climits>
#include <ctype.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
using namespace std;

/* Matrix size */
//Modify the following parameters
#define Nb 100 //number of bond grid points
#define Ny 100 //number of endowment grid points
#define max_iter 40 //number of samples collected
#define inner_loop 10 //number of inner loops of the component per sample

#define lbd -1
#define ubd 0
#define tol 1e-4
#define rstar 0.017
#define alpha 0.5
#define maxInd Ny*Nb
#define beta 0.953
#define phi 0.282
#define tau 0.5
#define delta 0.8
#define rho 0.9 
#define sigma 0.025

class Domain{
    public:

    std::vector<double> B;			       // bond grid
	std::vector<double> Y;
    std::vector<double> P;
    std::vector<double> V;
    std::vector<double> Price;
    std::vector<double> Vr;
    std::vector<double> Vd;
    std::vector<double> decision;

    std::vector<double> V0;
    std::vector<double> Vd0;
    std::vector<double> Price0;
    std::vector<double> prob;
    
    //constructor, store vectors
    Domain(std::vector<double> B2, std::vector<double> Y2, std::vector<double> P2, std::vector<double> V2,std::vector<double> Price2, std::vector<double> Vr2, std::vector<double> Vd2, std::vector<double> decision2, std::vector<double> prob2){
        B = B2;
        Y = Y2;
        P = P2;
        V = V2;
        Price = Price2;
        Vr = Vr2;
        Vd = Vd2;
        decision = decision2; 
        prob = prob2;
    }

    //retrieve vector elements
    //double& _ele(int index) {return [index];}
    //double& B_ele(int index) {return B[index];}
    double& B_ele(int index) {return B[index];}
    double& Y_ele(int index) {return Y[index];}
    double& P_ele(int index) {return P[index];}
    double& V_ele(int index) {return V[index];}
    double& Price_ele(int index) {return Price[index];}
    double& Vr_ele(int index) {return Vr[index];}
    double& Vd_ele(int index) {return Vd[index];}
    double& decision_ele(int index) {return decision[index];}
    double& V0_ele(int index) {return V0[index];}
    double& Vd0_ele(int index) {return Vd0[index];}
    double& Price0_ele(int index) {return Price0[index];}
    double& prob_ele(int index) {return prob[index];}

    void store_old(){
        V0 = V;
        Vd0 = Vd;
        Price0 = Price;
    }

    void print_all(){
        std::cout <<"printing decision" << std::endl;
        print_matrix(decision);
        std::cout <<"printing Price" << std::endl;
        print_matrix(Price);
        std::cout <<"printing Vr" << std::endl;
        print_matrix(Vr);
        std::cout <<"printing Vd" << std::endl;
        print_matrix(Vd);

    }

    void saxpy_series(){
        std::transform(std::execution::par,
                       V0.begin(), V0.end(), V.begin(), V.end(),
                       [=](double x, double y){return (1.0-delta)*x + delta*y;}
        );
        std::transform(std::execution::par,
                       Price0.begin(), Price0.end(), Price.begin(), Price.end(),
                       [=](double x, double y){return (1.0-delta)*x + delta*y;}
        );
        std::transform(std::execution::par,
                       Vd0.begin(), Vd0.end(), Vd.begin(), Vd.end(),
                       [=](double x, double y){return (1.0-delta)*x + delta*y;}
        );
    }

    double price_error(){
        double diff = 0.0;
        for (int i=0; i < Ny*Nb; i++){
            diff = std::max(abs(Price[i]-Price0[i]),diff);
        }
        return diff;
    }

    void print_matrix(std::vector<double>& M){
    for (std::vector<double>::const_iterator i=M.begin(); i!=M.end(); ++i)
    std::cout << *i <<' ';
    std::cout << std::endl;
    }


};


//=============================================================

//line 7
void default_value(Domain &domain, std::vector<int> vy){
    std::for_each(std::execution::par,
                    vy.begin(),vy.end(),
                    [&domain](int iy){
                    
                    double sumdef, consdef;
                    sumdef = 0.0;
                    consdef = pow(exp(domain.Y_ele(iy)*(1.0-tau)), (1.0-alpha))/(1.0-alpha);
                    for (int y = 0; y < Ny; y++){
                        sumdef += beta* domain.P_ele(iy + Ny*y) * (phi* domain.V0_ele(y) + (1.0-phi)*domain.Vd0_ele(y));
                    }
                    domain.Vd_ele(iy) = consdef + sumdef;
                  }  
    );
}

//line 8
void repayment_value(Domain &domain, std::vector<int> vby){
    std::for_each(std::execution::par,
                    vby.begin(),vby.end(),
                    [&domain](int index){
                    
                    int ib = index/(Ny);
                    int iy = index-Ny*ib;

                    double Max = -100000.0;
                    for (int b = 0; b < Nb; b++){
                        
                        double c = exp(domain.Y_ele(iy)) + domain.B_ele(ib) - domain.Price0_ele(iy+b*Ny)* domain.B_ele(b);
                        if (c > 0){
                            double sumret = 0.0;
                            for (int y = 0; y < Ny; y++){
                                sumret += domain.V0_ele(y+b*Ny) * domain.P_ele(iy+y*Ny);
                            };
                            double vr = pow(c, (1.0-alpha))/(1.0-alpha) + beta* sumret;
                            if (Max < vr) Max = vr;
                        }
                        
                    }
                    domain.Vr_ele(iy+ib*Ny) = Max;
                    } );
};

//line 9-14
void decide(Domain &domain, std::vector<int> vby){
    std::for_each(std::execution::par,
                    vby.begin(),vby.end(),
                    [&domain](int index){

                    int ib = index/(Ny);
                    int iy = index-Ny*ib;
                    
                    if (domain.Vd_ele(iy) < domain.Vr_ele(iy+ib*Ny)){
                        domain.V_ele(iy+ib*Ny) = domain.Vr_ele(iy+ib*Ny);
                        domain.decision_ele(iy+ib*Ny) = 0;
                    }
                    else {
                        domain.V_ele(iy+ib*Ny) = domain.Vd_ele(iy);
                        domain.decision_ele(iy+ib*Ny) = 1;
                    }

                    for (int y = 0; y < Ny; y++){
                        domain.prob_ele(iy+ib*Ny) += domain.P_ele(iy+y*Ny) * domain.decision_ele(y+ib*Ny);
                    }

                    domain.Price_ele(iy+ib*Ny) = (1.0-domain.prob_ele(iy+ib*Ny)) / (1.0+rstar);
                    }
    );

}



//===================================================================



// A stand alone normcdf
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
void tauchen(std::vector<double>& P){
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

//support function, thanks to Mike Seymour on Stacks Overflow
double median(vector<float> &v){
    size_t n = v.size()/2;
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

void print(std::vector <float> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < sizeof(a); i++)
   std::cout << a.at(i) << ' ';
   std::cout << endl;
}

int main(){
    //Initializing Bond matrix
    std::vector<double> B(Nb);
    double minB = lbd ; // minp*kss;
	double maxB = ubd ;  // maxp*kss;
	double step = (maxB-minB)/double(Nb-1);
	for (int i_b = 0; i_b < Nb; i_b++) {
		B[i_b] = minB + step*double(i_b);
	};

    //Intitializing Endowment matrix
	std::vector<double> Y(Ny);
    double sigma_z = sqrt(sigma*sigma/(1.0-rho*rho));
    double step2 = 10*sigma_z/(Ny-1);
    for (int i_y = 0; i_y < Ny; i_y++) {
		Y[i_y] = -5*sigma_z + step2*double(i_y);
	};


	
    std::vector<double> P(Ny*Ny,0.0);  //Conditional probability matrix
    std::vector<double> V(Ny*Nb,(1.0/((1.0-beta)*(1.0-alpha)))); //Value
    std::vector<double> Price(Ny*Nb, (1.0/(1.0+rstar))); //debt price
    std::vector<double> Vr(Ny*Nb, 0.0); //Value of good standing
    std::vector<double> Vd(Ny, 0.0); //Value of default
    std::vector<double> decision(Ny*Nb, 1.0); //Decision matrix
    std::vector<double> prob(Ny*Nb, 0.0);
    //Initialize Conditional Probability matrix
    tauchen(P);
    //check value of P
    

    //creating instance of domain
    Domain* domain;
    domain = new Domain(B,Y,P,V,Price,Vr,Vd,decision,prob);

    //main loop
    //int iter = 0;
    std::vector<int> vby(Nb*Ny);
    std::iota(std::begin(vby), std::end(vby), 0);

    std::vector<int> vy(Ny);
    std::iota(std::begin(vy), std::end(vy), 0);

    double err = 100.0;
    int iter = 0;
    vector<float> deftime;
    vector<float> repaytime;
    vector<float> decisiontime;
    vector<float> updatetime;
    
    while (iter<max_iter){
        std::cout << "starting iteration" << iter <<std::endl;
        (*domain).store_old();
        
        //std::cout << "default" <<std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < inner_loop; j++){
        default_value(*domain, vy);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        deftime.push_back(duration);

        //std::cout << "repayment" <<std::endl;
        t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < inner_loop; j++){
        repayment_value(*domain, vby);
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        repaytime.push_back(duration);
        //std::cout << "decide" <<std::endl;

        t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < inner_loop; j++){
        decide(*domain, vby);
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        decisiontime.push_back(duration);

        //std::cout << "calc error" <<std::endl;
        t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < inner_loop; j++){
        err = (*domain).price_error();
        //std::cout << "saxpy" <<std::endl;
        (*domain).saxpy_series();
        }
        t2 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
        updatetime.push_back(duration);
        //std::cout << "iter=" << iter << ", err=" << err <<std::endl;
        iter++;
    }
    //print(deftime);
    //print(repaytime);
    //print(decisiontime);
    //print(updatetime);

    double VD = median(deftime)/inner_loop;
    double VR = median(repaytime)/inner_loop;
    double DECIDE = median(decisiontime)/inner_loop;
    double UPDATE = median(updatetime)/inner_loop;
    std::cout << "deftime: " << VD << " ,repaytime: " << VR << " ,decisiontime: " << DECIDE <<" ,updatetime: " << UPDATE <<std::endl;
    std::cout << "Ny: " << Ny << "Nb: " << Nb <<std::endl;
    
    //(*domain).print_all();
    
}

