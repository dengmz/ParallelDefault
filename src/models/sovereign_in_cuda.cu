/* Includes, system */
// Solving sovereign default models using CUDA
// @Pablo A. Guerron
// Haverford, PA 2016

#include <iostream>
#include <iomanip>
#include <fstream>

// Includes, Thrust

#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

/* Includes, cuda */

#include <cublas_v2.h>


using namespace std;
using namespace thrust;

/* Matrix size */
#define nb 100
#define ny 7 
#define lbd -1
#define ubd 0
#define rrisk 2.0
#define bbeta 0.90
#define ppi 0.05
#define ttheta 0.282
#define tol 1e-6
#define rstar 0.017


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
void tauchen(double rrho, double ssigma, host_vector<double>& Z, host_vector<double>& P) {
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


void saxpy_fast(double A, device_vector<double>& X, device_vector<double>& Y)
{
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}


struct findmaxshort{
    
    double *B, *qrB, *Y, *EV, *vforcomp ;
    
    __host__ __device__
    findmaxshort(double *B_ptr, double *qrB_ptr, double *Y_ptr, double *EV_ptr, double *vforcomp_ptr){
        B = B_ptr; Y = Y_ptr ; qrB = qrB_ptr; Y = Y_ptr; EV = EV_ptr; vforcomp = vforcomp_ptr;   
    };
    
    __host__ __device__
    void operator()(int index){
        int i_b = index / (ny);
        int i_y = index - i_b*ny;
        double vlar = 0.0, vfirst = 0.0, sumdef = 0.0, sumbad = 0.0 ;
        double budget, consdef ;
        int left_ind = 0 ;
        int right_ind = nb ; // last two lines could be potentially erased
        int largest = left_ind, first = left_ind ;
        
        budget = B[i_b] + exp(Y[i_y]) ;

        while (first < right_ind) {

            (budget > qrB[i_y + ny*largest]) ? vlar = pow(budget - qrB[i_y + ny*largest], (1 - rrisk)) / (1 - rrisk) + bbeta * EV[i_y + ny*largest] : vlar = -10000000.0;
            (budget > qrB[i_y + ny*first]) ? vfirst = pow(budget - qrB[i_y + ny*first], (1 - rrisk)) / (1 - rrisk) + bbeta * EV[i_y + ny*first] : vfirst = -10000000.0;

            if (vlar < vfirst)
                largest = first;
            first++;
        }       
   
        vforcomp[i_y + ny*i_b] = vlar ;
        
    }        

};


struct decide{
    
    double *vforcomp, *Y, *value,  *wdef, *decision, *vbad, *P, *qprupd ;
    
    __host__ __device__
    decide(double *vforcomp_ptr, double *Y_ptr, double *value_ptr, double *wdef_ptr, double *decision_ptr, double *vbad_ptr, double *P_ptr, double *qprupd_ptr){
        vforcomp = vforcomp_ptr; Y = Y_ptr; value = value_ptr ; wdef = wdef_ptr; 
        decision = decision_ptr ; vbad = vbad_ptr; P = P_ptr, qprupd = qprupd_ptr ;        
    };
    
    __host__ __device__
    void operator()(int index){
        int i_b = index / (ny);
        int i_y = index - i_b*ny;
        double consdef, sumdef, sumbad ;
     
        int indn = i_y + ny*i_b ;
        
        if (vforcomp[indn] > wdef[i_y]){
            value[indn] = vforcomp[indn] ;
            decision[indn] = 0 ;
            // cout << i_y << " " << i_b << endl ;
        }
        else{
            value[indn] = wdef[i_y] ;
            decision[indn] = 1 ;
            // cout << "default " << iz << endl ; 
        }
        
       // Update prob of default
        sumdef = 0.0;
        for (int dfx = 0; dfx < ny; dfx++) sumdef += P[i_y+ny*dfx] * decision[dfx+ny*i_b];
        // Update debt price
        qprupd[indn] = (1.0 - sumdef) / (1 + rstar);
                
        sumbad = 0.0 ;
        consdef = pow(exp(Y[i_y])*(1-ppi),(1-rrisk))/(1-rrisk) ;
        for (int ttx = 0; ttx < ny; ttx++) sumbad += P[i_y+ny*ttx]*((1-ttheta)*vbad[ttx]+ttheta*value[ttx+ny*(nb-1)]) ;
        vbad[indn] = consdef + sumbad*bbeta ;         
        
    }
};


struct findmax{
    
    double *B, *qrB, *Y, *EV, *value,  *wdef, *decision, *vbad, *P, *qprupd ;
    
    __host__ __device__
    findmax(double *B_ptr, double *qrB_ptr, double *Y_ptr, double *EV_ptr, double *value_ptr, double *wdef_ptr, double *decision_ptr, double *vbad_ptr, double *P_ptr, double *qprupd_ptr){
        B = B_ptr; Y = Y_ptr ; qrB = qrB_ptr; Y = Y_ptr; EV = EV_ptr; value = value_ptr ; wdef = wdef_ptr; 
        decision = decision_ptr ; vbad = vbad_ptr; P = P_ptr, qprupd = qprupd_ptr ;    
    };
     
    __host__ __device__
    void operator()(int index){
        int i_b = index / (ny);
        int i_y = index - i_b*ny;
        double vlar = 0.0, vfirst = 0.0, sumdef = 0.0, sumbad = 0.0 ;
        double budget, consdef ;
        int left_ind = 0 ;
        int right_ind = nb ; // last two lines could be potentially erased
        int largest = left_ind, first = left_ind ;
        
        budget = B[i_b] + exp(Y[i_y]) ;

        while (first < right_ind) {

            (budget > qrB[i_y + ny*largest]) ? vlar = pow(budget - qrB[i_y + ny*largest], (1 - rrisk)) / (1 - rrisk) + bbeta * EV[i_y + ny*largest] : vlar = -10000000.0;
            (budget > qrB[i_y + ny*first]) ? vfirst = pow(budget - qrB[i_y + ny*first], (1 - rrisk)) / (1 - rrisk) + bbeta * EV[i_y + ny*first] : vfirst = -10000000.0;

            if (vlar < vfirst)
                largest = first;
            first++;
        }

        if (vlar > wdef[i_y]){
            value[index] = vlar ;
            decision[index] = 0 ;
        }
        else{
            value[index] = wdef[i_y] ;
            decision[index] = 1 ;
        }
        
       // Update prob of default
        sumdef = 0.0;
        for (int dfx = 0; dfx < ny; dfx++) sumdef += P[i_y+ny*dfx] * decision[dfx+ny*i_b];
        // Update debt price
        qprupd[index] = (1.0 - sumdef) / (1 + rstar);
        
        sumbad = 0.0 ;
        consdef = pow(exp(Y[i_y])*(1-ppi),(1-rrisk))/(1-rrisk) ;
        for (int ttx = 0; ttx < ny; ttx++) sumbad += P[i_y+ny*ttx]*((1-ttheta)*vbad[ttx]+ttheta*value[ttx+ny*(nb-1)]) ;
        vbad[index] = consdef + sumbad*bbeta ;           
    }        

};

// Parallelize summation
struct sumando{
    
    double *P, *value, *suma ;
  
    __host__ __device__
    sumando(double *P_ptr, double *value_ptr, double *suma_ptr){
        P = P_ptr ; value = value_ptr; suma = suma_ptr ;  
    };
    
    __host__ __device__
    void operator()(int index){
        int i_b = index / (ny) ;
        int i_y = index - i_b*ny ;
        double sumar = 0.0;

        for (int ttx = 0; ttx < ny; ttx++) {
            sumar += P[i_y + ny * ttx] * value[ttx + i_b * ny];
        }
        suma[index] = sumar ;
    }
    
};

struct sumando1{
    double *B, *qprice, *qprB;
    
    __host__ __device__
    sumando1(double *B_ptr, double *qprice_ptr, double *qprB_ptr){
        B = B_ptr ; qprice = qprice_ptr ; qprB = qprB_ptr ;
    }
    
    __host__ __device__
    void operator()(int index){
        int i_b = index/(ny) ;
        int i_y = index - i_b*ny;

        qprB[i_y + ny * i_b] = qprice[i_y + i_b * ny] * B[i_b];

    }
};

struct sumando2{
    double *P, *Y, *vbad, *wdef, *value;
    
    __host__ __device__
    sumando2(double *P_ptr, double *Y_ptr, double *vbad_ptr, double *wdef_ptr, double *value_ptr){
        P = P_ptr; Y = Y_ptr; vbad = vbad_ptr; wdef = wdef_ptr; value = value_ptr ;
    }
    
    __host__ __device__
    void operator()(int index){
        int i_b = index/(ny) ;
        int i_y = index - ny * i_b ;
        double sumdef, consdef ;

        sumdef = 0.0;
        consdef = pow(exp(Y[i_y])*(1 - ppi), (1 - rrisk)) / (1 - rrisk);
        // for (int ttx = 0; ttx < ny; ttx++) sumdef += P[i_y + ny * ttx] * vbad[ttx]; // old incorrect value
        for (int ttx = 0; ttx < ny; ttx++) sumdef += P[i_y + ny * ttx] * ((1-ttheta)*vbad[ttx]+ttheta*value[ttx+ny*(nb-1)]) ;
        wdef[i_y] = consdef + bbeta*sumdef;       
        
    }
};

// This functor calculates the error
struct myMinus {
	template <typename Tuple>
	__host__ __device__
	double operator()(Tuple t)
	{
		return abs(get<0>(t)-get<1>(t));
	}
};

int main(void){
  
        double alpha = 1 , beta = 0 ;
        double err = 10 ;
        double err1 = 10 ;
        const double lamupd = 0.99 ;
        const double lampudp = 1.0 - lamupd ;

	host_vector<double> h_B(nb);			       // bond grid
	host_vector<double> h_Y(ny);			       // endowment grid
	host_vector<double> h_P(ny*ny,0.0);
        host_vector<double> h_qprice(ny*nb,1.0/(1+rstar));
        host_vector<double> h_qprupd(ny*nb,1.0/(1+rstar));
        host_vector<double> h_qtemp(ny*nb,1.0/(1+rstar));
        host_vector<double> h_vbad(ny*nb,0.0), h_vbadold(ny*nb,0.0), h_value(ny*nb,1.0/((1-bbeta)*(1-rrisk))) ;
        host_vector<double> h_vtemp(ny*nb), h_valueold(ny*nb) ;
        host_vector<double> h_wndef(ny*nb), h_wdef(ny,0.0), h_locoptndef(ny*nb,0.0), h_decision(ny*nb,1.0) ;
        host_vector<double> h_probdef(ny*nb,1.0) ;
        host_vector<double> h_wtemp(ny*nb,-1000.0) ;
        host_vector<double> h_qprB(ny*nb,0.0) ;
        host_vector<double> h_suma(ny*nb,0.0) ;
        host_vector<double> h_vforcomp(ny*nb,0.0) ;
	
        // Initialize capital grid
	double minB = lbd ; // minp*kss;
	double maxB = ubd ;  // maxp*kss;
	double step = (maxB-minB)/double(nb-1);
	for (int i_b = 0; i_b < nb; i_b++) {
		h_B[i_b] = minB + step*double(i_b);
	};

	// initialize shock grid
        double rrho = 0.9 ;
        double ssigma = 0.025 ;
        tauchen(rrho, ssigma, h_Y, h_P); // discretizing a N(0,1) as epsilon

	// cout << h_P[1+1*ny] << endl;        
        
        // Initialize pointers 
        double* h_B_ptr = raw_pointer_cast(h_B.data());
        double* h_Y_ptr = raw_pointer_cast(h_Y.data());
        double* h_P_ptr = raw_pointer_cast(h_P.data());
        double* h_qprice_ptr = raw_pointer_cast(h_qprice.data());
        double* h_qprupd_ptr = raw_pointer_cast(h_qprupd.data());
        double* h_qtemp_ptr = raw_pointer_cast(h_qtemp.data());
        double* h_vbad_ptr = raw_pointer_cast(h_vbad.data());
        double* h_vbadold_ptr = raw_pointer_cast(h_vbadold.data());
        double* h_value_ptr = raw_pointer_cast(h_value.data());
        double* h_vtemp_ptr = raw_pointer_cast(h_vtemp.data());
        double* h_valueold_ptr = raw_pointer_cast(h_valueold.data());
        double* h_wndef_ptr = raw_pointer_cast(h_wndef.data());
        double* h_wdef_ptr = raw_pointer_cast(h_wdef.data());
        double* h_locoptndef_ptr = raw_pointer_cast(h_locoptndef.data());
        double* h_decision_ptr = raw_pointer_cast(h_decision.data());
        double* h_wtemp_ptr = raw_pointer_cast(h_wtemp.data());
        double* h_probdef_ptr = raw_pointer_cast(h_probdef.data());
        double* h_qprB_ptr = raw_pointer_cast(h_qprB.data()) ;
        double* h_suma_ptr = raw_pointer_cast(h_suma.data()) ;
        double* h_vforcomp_ptr = raw_pointer_cast(h_vforcomp.data()) ;
        
        // Copy to device        
	device_vector<double> d_B = h_B;			       // bond grid
	device_vector<double> d_Y = h_Y;			       // endowment grid
	device_vector<double> d_P = h_P;
        device_vector<double> d_qprice= h_qprice;
        device_vector<double> d_qprupd = h_qprupd;
        device_vector<double> d_qtemp = h_qtemp;
        device_vector<double> d_vbad = h_vbad;
        device_vector<double> d_vbadold = h_vbadold; 
        device_vector<double> d_value = h_value ;
        device_vector<double> d_vtemp = h_vtemp;
        device_vector<double> d_valueold= h_valueold ;
        device_vector<double> d_wndef = h_wndef; 
        device_vector<double> d_wdef = h_wdef; 
        device_vector<double> d_locoptndef = h_locoptndef;
        device_vector<double> d_decision = h_decision ;
        device_vector<double> d_probdef = h_probdef ;
        device_vector<double> d_wtemp = h_wtemp ;
        device_vector<double> d_qprB = h_qprB ;
        device_vector<double> d_suma = h_suma ;
        device_vector<double> d_vforcomp = h_vforcomp ;
        
        // Pointers to device
        double* d_B_ptr = raw_pointer_cast(d_B.data());
        double* d_Y_ptr = raw_pointer_cast(d_Y.data());
        double* d_P_ptr = raw_pointer_cast(d_P.data());
        double* d_qprice_ptr = raw_pointer_cast(d_qprice.data());
        double* d_qprupd_ptr = raw_pointer_cast(d_qprupd.data());
        double* d_qtemp_ptr = raw_pointer_cast(d_qtemp.data());
        double* d_vbad_ptr = raw_pointer_cast(d_vbad.data());
        double* d_vbadold_ptr = raw_pointer_cast(d_vbadold.data());
        double* d_value_ptr = raw_pointer_cast(d_value.data());
        double* d_vtemp_ptr = raw_pointer_cast(d_vtemp.data());
        double* d_valueold_ptr = raw_pointer_cast(d_valueold.data());
        double* d_wndef_ptr = raw_pointer_cast(d_wndef.data());
        double* d_wdef_ptr = raw_pointer_cast(d_wdef.data());
        double* d_locoptndef_ptr = raw_pointer_cast(d_locoptndef.data());
        double* d_decision_ptr = raw_pointer_cast(d_decision.data());
        double* d_wtemp_ptr = raw_pointer_cast(d_wtemp.data());
        double* d_probdef_ptr = raw_pointer_cast(d_probdef.data());   
        double* d_qprB_ptr = raw_pointer_cast(d_qprB.data()) ;
        double* d_suma_ptr = raw_pointer_cast(d_suma.data()) ;
        double* d_vforcomp_ptr = raw_pointer_cast(d_vforcomp.data()) ;
        
        
	// Firstly a virtual index array from 0 to nb*ny
	counting_iterator<int> begin(0);
	counting_iterator<int> end(nb*ny);

	// Step.1 Has to start with this command to create a handle
	cublasHandle_t handle;

	// Step.2 Initialize a cuBLAS context using Create function,
	// and has to be destroyed later
	cublasCreate(&handle);
      
        // Some definitions
        int iter = 0 ;
        
        cout << setprecision(8) ;
        
        // Main loop
        
        while (err>tol && iter < 5000){
            
            d_valueold = d_value ;
            d_vbadold  = d_vbad ;
            
            // Compute valuation of debt holdings
            thrust::for_each(begin,end,sumando1(d_B_ptr,d_qprice_ptr, d_qprB_ptr)) ;
            
            // Compute value of default
            thrust::for_each(begin,end,sumando2(d_P_ptr, d_Y_ptr, d_vbad_ptr, d_wdef_ptr,d_value_ptr)) ;
            
            // Find expected value: EV = V*P'
        	// C = alpha*op(A)*op(B) + beta*C;
           // /*
		cublasDgemm(handle,
				CUBLAS_OP_N,  
				CUBLAS_OP_N,
				ny, nb, ny, // op(A) is nk by nz, op(B) is nz by nz
				&alpha, //
				d_P_ptr, // pointer to A 
				ny, // # of row in A, not op(A)
				d_value_ptr,
				ny, // # of row in B, not op(B)
				&beta,
				d_suma_ptr, //
				ny); // # of row in C
            
            // */
            
             
            // Find max and decision rule
            // Solve everything in one block
            
            thrust::for_each(
                    begin,
                    end,
                    findmax(d_B_ptr, d_qprB_ptr, d_Y_ptr, d_suma_ptr, d_value_ptr, d_wdef_ptr, d_decision_ptr, d_vbad_ptr, d_P_ptr, d_qprupd_ptr)); 
            
            // Split findmax 
            /*
            thrust::for_each(
                    begin,
                    end,
                    findmaxshort(d_B_ptr, d_qprB_ptr, d_Y_ptr, d_suma_ptr, d_vforcomp_ptr));             

             thrust::for_each(
                    begin,
                    end,
                    decide(d_vforcomp_ptr, d_Y_ptr, d_value_ptr, d_wdef_ptr, d_decision_ptr, d_vbad_ptr, d_P_ptr, d_qprupd_ptr));             
            */

            // Find error
            err = transform_reduce(
                    make_zip_iterator(make_tuple(d_valueold.begin(), d_value.begin())),
                    make_zip_iterator(make_tuple(d_valueold.end(), d_value.end())),
                    myMinus(), // abs(V[i]-Vplus[i]) => vector
                    0.0, // first find max (0, diff[0]), then 
                    maximum<double>()
                    );
            
             err1 = transform_reduce(
                    make_zip_iterator(make_tuple(d_qprice.begin(), d_qprupd.begin())),
                    make_zip_iterator(make_tuple(d_qprice.end(), d_qprupd.end())),
                    myMinus(), // abs(V[i]-Vplus[i]) => vector
                    0.0, // first find max (0, diff[0]), then 
                    maximum<double>()
                    );           
            
            err = max(err,err1) ;
    
            cout << "Errors in iteration: " << iter << " is " << err << endl ;
            
            iter++ ;
            
            // Update price and value functions
            
            saxpy_fast(lampudp, d_qprupd, d_qprice) ;
            saxpy_fast(lamupd, d_valueold, d_value) ;
            saxpy_fast(lamupd, d_vbadold, d_vbad) ;
            
            
        };

	// Step.3 Destroy the handle.
	cublasDestroy(handle);
        
        
        // Save results
        ofstream policy_out, value_out, price_out, decision_out;
        policy_out.open("qprice.csv"); // open file to print optimal
        value_out.open("vopt.csv"); // open file to print optimal
        price_out.open("moreprice.csv");
        decision_out.open("decision.csv");

        int indec = ny*nb ;
        for (int i_e = 0; i_e < indec; i_e++) {
            policy_out << d_qprice[i_e] << '\n';
            value_out << d_value[i_e] << '\n';
            decision_out << d_decision[i_e] << '\n';
            // cout << qprice[i_e] << endl;
        }

        for (device_vector<double>::iterator it = d_qprice.begin(); it != d_qprice.end(); it++) {
            price_out << *it << '\n';
        }

        // finish saving
        policy_out.close();
        value_out.close();
        price_out.close();
        decision_out.close();


	return 0;
}

