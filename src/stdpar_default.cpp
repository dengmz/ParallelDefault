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
#include <fstream>
#include <chrono>

/* Matrix size */
#define Nb 100
#define Ny 500
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


//main data structure, inspired by data structure of LULESH, adding details later

class Domain{

    public:
    std::vector<double> B; //Conditional probability matrix
	std::vector<double> Y;
    std::vector<double> P;
    std::vector<double> V; //Value
    std::vector<double> Price; //debt price
    std::vector<double> Vr; //Value of good standing
    std::vector<double> Vd; //Value of default
    std::vector<double> decision; //Decision matrix

    std::vector<double> V0; //old value
    std::vector<double> Vd0; //old default value
    std::vector<double> Price0; //old price
    std::vector<double> prob; //prob matrix
    
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

    //retrieve vector elements in stdpar executions
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

    //store last round values
    void store_old(){
        V0 = V;
        Vd0 = Vd;
        Price0 = Price;
    }

    //testing print
    void print_all(){
        std::cout <<"printing decision" << std::endl;
        print_matrix(decision, Ny);
        std::cout <<"printing Price" << std::endl;
        print_matrix(Price, Ny);
        std::cout <<"printing Vr" << std::endl;
        print_matrix(Vr, Ny);
        std::cout <<"printing Vd" << std::endl;
        print_matrix(Vd, Ny);

    }

    //saxpy operation for line 16
    void saxpy_series(std::vector<int> vby, std::vector<int> vy){
        
        std::for_each(std::execution::par,
                       vby.begin(),vby.end(),
                       [=](int index){V[index] = (1.0-delta)*V0[index] + delta*V[index];}
        );
        std::for_each(std::execution::par,
                       vby.begin(),vby.end(),
                       [=](int index){Price[index] = (1.0-delta)*Price0[index] + delta*Price[index];}
        );
        std::for_each(std::execution::par,
                       vy.begin(),vy.end(),
                       [=](int index){Vd[index] = (1.0-delta)*Vd0[index] + delta*Vd[index];}
        );
        
    }

    //calcualte price error
    double price_error(){
        double diff = 0.0;
        for (int i=0; i < Ny*Nb; i++){
            diff = std::max(abs(Price[i]-Price0[i]),diff);
        }
        return diff;
    }

    //calcualte value error
    double value_error(){
        double diff = 0.0;
        for (int i=0; i < Ny*Nb; i++){
            diff = std::max(abs(V[i]-V0[i]),diff);
        }
        return diff;
    }

    void print_matrix(std::vector<double>& M, int line){
        int counter = 0;
        for (std::vector<double>::const_iterator i=M.begin(); i!=M.end(); ++i){
            std::cout << *i <<' ';
            counter++;
            if (counter%line == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};


//=============================================================
//Main Calculation processes
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

//===================================================================

int main(){
    //Initializing Bond matrix
    auto start = std::chrono::high_resolution_clock::now();

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


    std::vector<int> vby(Nb*Ny);
    std::iota(std::begin(vby), std::end(vby), 0);

    std::vector<int> vy(Ny);
    std::iota(std::begin(vy), std::end(vy), 0);
    
    //main loop
    double err = 100.0;
    int iter = 1;
    
    while (err > tol && iter<1000){
        
        std::cout << "=====starting iteration=====" <<std::endl;
        (*domain).store_old();
        
        //std::cout << "=====default=====" <<std::endl;
        default_value(*domain, vy);
        //std::cout << "=====repayment=====" <<std::endl;
        repayment_value(*domain, vby);
        //std::cout << "=====decide=====" <<std::endl;
        (*domain).prob = prob;
        decide(*domain, vby);

        //std::cout << "=====calc error=====" <<std::endl;
        err = (*domain).value_error();
        //std::cout << "=====saxpy=====" <<std::endl;
        (*domain).saxpy_series(vby,vy);
        std::cout << "=====iter=" << iter << ", err=" << err <<std::endl;
        //(*domain).print_all();
        
       iter++;
    }
    
    //(*domain).print_all();
    //recording time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
    std::cout<<"number of microseconds: " << duration.count()<<std::endl;

    // Save results
    std::ofstream value_out, price_out, decision_out;
    value_out.open("my_value.csv"); // open file to print optimal
    price_out.open("my_my_price.csv");
    decision_out.open("my_decision.csv");


    for (int i = 1; i < (Ny*Nb+1); i++) {
        if(i%Ny == 0){
            value_out << '\n';
            price_out << '\n';
            decision_out << '\n';
        }
        else{
            value_out << domain->V_ele(i) << ',';
            price_out << domain->Price_ele(i) << ',';
            decision_out << domain->decision_ele(i) << ',';
        }
        
    }
    
    value_out.close();
    price_out.close();
    decision_out.close();

    return 0;
    
}
