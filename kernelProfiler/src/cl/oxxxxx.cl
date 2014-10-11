#include "cmplx.h"
_CL_CUDA_DEVICE_
void oxxxxx(double* p, double fmass, int nhel, int nsf,
	    cmplx* fo)
{
  fo[4] = mkcmplx(p[0]*nsf, p[3]*nsf);
  fo[5] = mkcmplx(p[1]*nsf, p[2]*nsf);
  int nh = nhel*nsf;
  
  cmplx chi[2];
  
  if (fmass!=0.) {
    double pp = fmin(p[0],
		     sqrt(p[1]*p[1] + p[2]*p[2] + p[3]*p[3]));
    
    if (pp==0.) {
      double sqm[2];
      
      sqm[0] = sqrt(fabs(fmass));
      
      sqm[1] = copysign(sqm[0], fmass);
      
      int ip = -(1+nh)/2;
      
      int im = (1-nh)/2;
      
      fo[0] = mkcmplx((double)(im) *sqm[im]);
      fo[1] = mkcmplx((double)(ip*nsf)*sqm[im]);
      fo[2] = mkcmplx((double)(im*nsf)*sqm[-ip]);
      fo[3] = mkcmplx((double)(ip) *sqm[-ip]);
      
    } else {
      double sf[2],omega[2];
      
      sf[0] = (double)(1 + nsf + (1-nsf)*nh)*0.5;
      sf[1] = (double)(1 + nsf - (1-nsf)*nh)*0.5;
      omega[0] = sqrt(p[0]+pp);
      
      omega[1] = fmass*(1./omega[0]);
      
      double pp3 = fmax(pp+p[3],0.);
      
      chi[0] = mkcmplx(sqrt(pp3*0.5*(1./pp)));
      if (pp3==0.) {
	chi[1] = mkcmplx((double)(nh));
      } else {
	chi[1] = rsqrt(2.*pp*pp3) * mkcmplx((double)(nh)*p[1],-p[2]);
	
      }
      int ip = (3+nh)/2-1;
      
      int im = (3-nh)/2-1;
      
      fo[0] = sf[1]*omega[im]*chi[im];
      fo[1] = sf[1]*omega[im]*chi[ip];
      fo[2] = sf[0]*omega[ip]*chi[im];
      fo[3] = sf[0]*omega[ip]*chi[ip];
      
    }
  } else {
    double sqp0p3;
    
    if (p[1]==0. && p[2]==0. && p[3]<0.) {
      sqp0p3 = 0.;
    } else {
      sqp0p3 = sqrt(fmax(p[0]+p[3],0.))*(double)(nsf);
    }
    chi[0] = mkcmplx(sqp0p3);
    if (sqp0p3==0.) {
      chi[1] = mkcmplx((double)(nhel)*sqrt(2.*p[0]));
    } else {
      chi[1] = (1./sqp0p3) * mkcmplx((double)(nh)*p[1],-p[2]);
      
    }
    cmplx czero = mkcmplx(0.,0.);
    if (nh==1) {
      fo[0] = chi[0];
      fo[1] = chi[1];
      fo[2] = czero;
      fo[3] = czero;
      
    } else {
      fo[0] = czero;
      fo[1] = czero;
      fo[2] = chi[1];
      fo[3] = chi[0];
      
    } }
}
