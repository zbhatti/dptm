#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define cmplx a_cmplx_t

typedef struct a_cmplx_t {
	float re; 
	float im;
} cmplx;


void oxxxxx(double* p, double fmass, int nhel, int nsf, cmplx* fo){
	
	cmplx tst = {124.1, 124.412};
	fo[4].re = p[0]*nsf; 
	fo[4].im = p[3]*nsf;
	
	fo[5].re = p[1]*nsf; 
	fo[5].re=p[2]*nsf;
	
	int nh = nhel*nsf;
	cmplx chi[2];

	if (fmass!=0.) {
		double pp = fmin(p[0],sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3] )  );

		if (pp==0.) {
			double sqm[2];

			sqm[0] = sqrt(fabs(fmass) );

			sqm[1] = copysign(sqm[0], fmass);

			int ip = -(1+nh)/2;

			int im = (1-nh)/2;

			fo[0].re = (double)(im) *sqm[im];
			fo[0].im = 0;
			
			fo[1].re = (double)(ip*nsf)*sqm[im];
			fo[1].im = 0;
			
			fo[2].re = (double)(im*nsf)*sqm[-ip];
			fo[2].im = 0;
			
			fo[3].re = (double)(ip) *sqm[-ip];
			fo[3].im = 0;

		} 
	
		else {
			double sf[2],omega[2];

			sf[0] = (double)(1 + nsf + (1-nsf)*nh)*0.5;
			sf[1] = (double)(1 + nsf - (1-nsf)*nh)*0.5;
			omega[0] = sqrt(p[0]+pp);

			omega[1] = fmass*(1./omega[0]);

			double pp3 = fmax(pp+p[3],0.);

			chi[0].re = sqrt(pp3*0.5*(1./pp));
			chi[0].im = 0;
			
			if (pp3==0.) {
				chi[1].re = (double)(nh);
				chi[1].im = 0;
			} 
			
			else {
				chi[1].re =  (double)(nh)*p[1] / sqrt(2.*pp*pp3)  ;
				chi[1].im =  -p[2]/sqrt(2.*pp*pp3)  ;
			}
			int ip = (3+nh)/2-1;

			int im = (3-nh)/2 - 1;

			fo[0].re = sf[1]*omega[im]*chi[im].re;
			fo[0].im = sf[1]*omega[im]*chi[im].im;
			
			fo[1].re = sf[1]*omega[im]*chi[ip].re;
			fo[1].im = sf[1]*omega[im]*chi[ip].im;
			
			fo[2].re = sf[0]*omega[ip]*chi[im].re;
			fo[2].im = sf[0]*omega[ip]*chi[im].im;
			
			fo[3].re = sf[0]*omega[ip]*chi[ip].re;
			fo[3].im = sf[0]*omega[ip]*chi[ip].im;
		}
	
	} 
	
	else {
		double sqp0p3;

		if (p[1]==0. && p[2]==0. && p[3]<0.) {
			sqp0p3 = 0.;
		} 
		else {
			sqp0p3 = sqrt(fmax(p[0]+p[3],0.))*(double)(nsf);
		}
		chi[0].re = sqp0p3;
		chi[0].im = 0;
		
		if (sqp0p3==0.) {
			chi[1].re = (double)(nhel)*sqrt(2.*p[0]);
			chi[1].im = 0;
		} 
		
		else {
			chi[1].re = (1./sqp0p3) * (double)(nh)*p[1];
			chi[1].im = (1./sqp0p3) * (-p[2]);
		}
		
		cmplx czero = {0., 0.};

		if (nh==1) {
			fo[0] = chi[0];
			fo[1] = chi[1];
			fo[2] = czero;
			fo[3] = czero;

		} 
		
		else {
			fo[0] = czero;
			fo[1] = czero;
			fo[2] = chi[1];
			fo[3] = chi[0];
		} 
	
	}
	
}

__kernel
void Oxxxxx(__global const double * P_d, __global cmplx * Fo_d, const int Psize)
{
	int idx = get_global_id(0);
	double fmass = 124.412;
	int nhel = 2;
	int nsf = 4;
	
	double P[4];
	cmplx Fo[6];
	
	P[0] = P_d[idx*4 + 0];
	P[1] = P_d[idx*4 + 1];
	P[2] = P_d[idx*4 + 2];
	P[3] = P_d[idx*4 + 3];
	
	if (idx*4 < Psize){
		oxxxxx(P, fmass, nhel, nsf, Fo);
	}
	
	else
		return;
	
	//synch?
	Fo_d[6*idx + 0] = Fo[0];
	Fo_d[6*idx + 1] = Fo[1];
	Fo_d[6*idx + 2] = Fo[2];
	Fo_d[6*idx + 3] = Fo[3];
	Fo_d[6*idx + 4] = Fo[4];
	Fo_d[6*idx + 5] = Fo[5];
	
}
