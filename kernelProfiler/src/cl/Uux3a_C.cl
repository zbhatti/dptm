#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define rSQRT2 0.707106781186
#define cmplx a_cmplx_t

typedef struct a_cmplx_t {
	float re; 
	float im;
} cmplx;

a_cmplx_t mkcmplx(float x, float y) {
	return (cmplx) {x,y};
}

a_cmplx_t ADD(a_cmplx_t X, a_cmplx_t Y) {
	return (cmplx) {X.re + Y.re, X.im + Y.im}; 
}
a_cmplx_t SUB(a_cmplx_t X, a_cmplx_t Y) {
	return (cmplx) {X.re - Y.re, X.im - Y.im}; 
}
a_cmplx_t SMUL(int a, a_cmplx_t X){
	return (cmplx) {a* X.re, a* X.im}; 
}
a_cmplx_t MUL(a_cmplx_t a, a_cmplx_t b){ 
	return (cmplx) {(a.re * b.re) - (a.im * b.im), (a.re * b.im) + (a.im * b.re)}; 
}
a_cmplx_t conj(a_cmplx_t a ) { 
	return (cmplx) {a.re,-a.im};
} 

void ixxxx1(__global const float* p, int nHEL, int nSF, cmplx* fi)
{
	float p0 = p[0];
	float p1 = p[1];
	float p2 = p[2];
	float p3 = p[3];
	
	float SQP0P3 = sqrt(p0+p3)*(float)(nSF);
	int NH = nHEL*nSF;
	fi[4] = mkcmplx(p0*(float)(nSF), p3*(float)(nSF));
	fi[5] = mkcmplx(p1*(float)(nSF), p2*(float)(nSF));
	
	cmplx CHI = mkcmplx(NH*p[1]*(1.0f/SQP0P3), p2*(1.0f/SQP0P3));
	cmplx CZERO = mkcmplx(0.0f, 0.0f);
	cmplx CSQP0P3 = mkcmplx(SQP0P3, 0.0f);
	
	fi[0] = ADD(SMUL((NH== 1),CZERO), SMUL((NH==-1),CHI));
	fi[1] = ADD(SMUL((NH== 1),CZERO), SMUL((NH==-1),CSQP0P3));
	fi[2] = ADD(SMUL((NH== 1),CSQP0P3), SMUL((NH==-1),CZERO));
	fi[3] = ADD(SMUL((NH== 1),CHI), SMUL((NH==-1),CZERO));
	return;
}


void oxxxx2(__global const float* p, int nHEL, int nSF, cmplx* fo)
	{
	
	float p0 = p[0];
	float p1 = p[1];
	float p2 = p[2];
	float p3 = p[3];
	
	int NH=nHEL*nSF;
	fo[4] = mkcmplx(p0*(float)(nSF), p3*(float)(nSF));
	fo[5] = mkcmplx(p1*(float)(nSF), p2*(float)(nSF));
	cmplx CHI = mkcmplx(-nHEL*sqrt(2.0f*p0), 0.0f);
	cmplx CZERO = mkcmplx(0.0f,0.0f);
	fo[0]=CZERO;
	fo[1]=ADD(SMUL((NH== 1),CHI), SMUL((NH==-1),CZERO));
	fo[2]=ADD(SMUL((NH== 1),CZERO), SMUL((NH==-1),CHI));
	fo[3]=CZERO;
	return;
}


void vxxxx0(__global const float* p, int nHEL, int nSV, cmplx* vc){

	float p0 = p[0];
	float p1 = p[1];
	float p2 = p[2];
	float p3 = p[3];

	vc[4] = SMUL( nSV, mkcmplx(p0,p3) );
	vc[5] = SMUL( nSV, mkcmplx(p1, p2) );
	float rpt = 1/sqrt(p1*p1 + p2*p2);
	vc[0] = mkcmplx(0.0f, 0.0f);
	vc[3] = mkcmplx( (float)(nHEL)*(1.0f/(rpt*p0))*rSQRT2, 0.0f);
	float pzpt = (p3*(1.0f/p0)*rpt)*rSQRT2 *(float)(nHEL);
	vc[1] = mkcmplx(-p1*pzpt, -nSV*p2 * rpt * rSQRT2);
	vc[2] = mkcmplx(-p2*pzpt,	+nSV*p1 * rpt * rSQRT2);
	return;
}


void fvoxx0(cmplx* fo, cmplx* vc, float* gal, cmplx* fvo)
{
	fvo[4] = ADD(fo[4],vc[4]);
	fvo[5] = ADD(fo[5],vc[5]);
	float pf[4];
	pf[0] = fvo[4].re;
	pf[1] = fvo[5].re;
	pf[2] = fvo[5].im;
	pf[3] = fvo[4].im;
	float pf2 = pf[0]*pf[0] - (pf[1]*pf[1] + pf[2]*pf[2] + pf[3]*pf[3]);
	cmplx cI = mkcmplx( 0.0f, 1.0f);
	float d = -1.0f/pf2;
	cmplx sl1 = ADD(MUL(ADD(vc[0],vc[3]),fo[2]), MUL(ADD(vc[1],MUL(cI,vc[2])),fo[3]));
	cmplx sl2 = ADD(MUL(SUB(vc[0],vc[3]),fo[3]) , MUL(SUB(vc[1], MUL(cI,vc[2])),fo[2]));
	cmplx sr1 = SUB(MUL(SUB(vc[0], vc[3]),fo[0]), MUL(ADD(vc[1],MUL(cI,vc[2])),fo[1]));
	cmplx sr2 = SUB(MUL(ADD(vc[0],vc[3]),fo[1]), MUL(SUB(vc[1],MUL(cI,vc[2])),fo[0]));
	
	fvo[0] = SMUL(d, SMUL( gal[1], ADD( SMUL(pf[0]+pf[3],sr1), MUL(fvo[5],sr2))));
	fvo[1] = SMUL(d, SMUL( gal[1], ADD( SMUL(pf[0]-pf[3],sr2), MUL(conj(fvo[5]),sr1))));
	fvo[2] = SMUL(d, SMUL( gal[0], SUB( SMUL(pf[0]-pf[3],sl1), MUL(fvo[5],sl2))));
	fvo[3] = SMUL(d, SMUL( gal[0], SUB( SMUL(pf[0]+pf[3],sl2), MUL(conj(fvo[5]),sl1))));
	return;
}

//note: this was defined as iovxx0() in http://arxiv.org/pdf/0908.4403.pdf pg 12

void iovxxx(cmplx* fi, cmplx* fo, cmplx* vc, float* gal, cmplx* vertex)
{

vertex[0] =
	ADD(
	SMUL(gal[0],
	SUB(ADD(MUL(ADD(MUL(fo[2], fi[0]),MUL(fo[3],fi[1])),vc[0]),
	MUL(ADD(MUL(fo[2],fi[1]), MUL(fo[3],fi[0])),vc[1])),
	ADD(MUL(MUL(SUB(MUL(fo[2],fi[1]), MUL(fo[3],fi[0])),vc[2]), mkcmplx(0.0f, 1.0f)), 
	MUL(SUB(MUL(fo[2],fi[0]), MUL(fo[3],fi[1]	)	), vc[3]	)))	)
	
	,
	
	SMUL(gal[1],(
	SUB(ADD(SUB(MUL( ADD(MUL(fo[0],fi[2]), MUL(fo[1],fi[3]) ), vc[0]),
	MUL( ADD( MUL(fo[0],fi[3]), MUL(fo[1],fi[2])), vc[1])),
	MUL(MUL(SUB(MUL(fo[0],fi[3]),MUL(fo[1],fi[2])),vc[2]), mkcmplx(0.0f, 1.0f))),
	MUL(SUB(MUL(fo[0],fi[2]), MUL(fo[1],fi[3])),vc[3]))	))
	);
	
return;
}


void fvixx0(cmplx* fi, cmplx* vc, float* gal, cmplx* fvi)
	{
	fvi[4] = SUB(fi[4],vc[4]);
	fvi[5] = SUB(fi[5],vc[5]);
	float pf[4];
	pf[0] = fvi[4].re;
	pf[1] = fvi[5].re;
	pf[2] = fvi[5].im;
	pf[3] = fvi[4].im;
	float pf2 = pf[0]*pf[0] -  (pf[1]*pf[1] + pf[2]*pf[2] + pf[3]*pf[3]);
	cmplx cI = mkcmplx( 0.0f, 1.0f);
	float d = -1.0f/pf2;
	cmplx sl1 = ADD(MUL(ADD(vc[0], vc[3]),fi[0]), MUL(SUB(vc[1], MUL(cI,vc[2])),fi[1]));
	cmplx sl2 = ADD(MUL(SUB(vc[0], vc[3]),fi[1]), MUL(ADD(vc[1], MUL(cI,vc[2])),fi[0]));
	cmplx sr1 = SUB(MUL(SUB(vc[0], vc[3]),fi[2]), MUL(SUB(vc[1], MUL(cI,vc[2])),fi[3]));
	cmplx sr2 = SUB(MUL(ADD(vc[0], vc[3]),fi[3]), MUL(ADD(vc[1], MUL(cI,vc[2])),fi[2]));
	
	fvi[0] = SMUL(d, SMUL( gal[0],  SUB(SMUL(pf[0]-pf[3],sl1), MUL(conj(fvi[5]),sl2))  ));
	fvi[1] = SMUL(d, SMUL( gal[0],  SUB(SMUL(pf[0]+pf[3],sl2), MUL(fvi[5],sl1))  ));
	fvi[2] = SMUL(d, SMUL( gal[1],  ADD(SMUL(pf[0]+pf[3],sr1), MUL(conj(fvi[5]),sr2))  ));
	fvi[3] = SMUL(d, SMUL( gal[1],  ADD(SMUL(pf[0]-pf[3],sr2), MUL(fvi[5],sr1))  ));
	return;
}

// Each thread corresponds to an event
// Each thread has 5 particles that have a 4momentum description
// No ouput

__kernel void Uux3a(__global const float *P_d, __global cmplx* Amp_d, int nEvents){
	
	int idx = get_global_id(0);
	
	if (idx > nEvents)
		return;
	
	//first term gets us to the correct event in P_d, the second one gets us the corresponding 4momentum for each particle
	__global const float *p1 = &P_d[idx*5*4 + 4*0];
	__global const float *p2 = &P_d[idx*5*4 + 4*1];
	__global const float *p3 = &P_d[idx*5*4 + 4*2];
	__global const float *p4 = &P_d[idx*5*4 + 4*3];
	__global const float *p5 = &P_d[idx*5*4 + 4*4];
	
	// coupling constants of FFV vertex, using meaningless fillers
	float gau[2];
	gau[0] = 1.;
	gau[1] = 1.;
	
	//twice fermion helicity (-1 or 1), using meaningless fillers
	int nh1 = -1;
	int nh2 = 1;
	int nh3 = -1;
	int nh4 = -1;
	int nh5 = 1;
	
	cmplx w01[6], w02[6], w03[6], w04[6], w05[6];
	ixxxx1(p1, nh1, +1, w01);
	oxxxx2(p2, nh2, -1, w02);
	vxxxx0(p3, nh3, +1, w03);
	vxxxx0(p4, nh4, +1, w04);
	vxxxx0(p5, nh5, +1, w05);
	
	cmplx w06[6], w07[6], w08[6];
	cmplx ampsum = mkcmplx(0.0f, 0.0f);
	cmplx amp;
	
	/*
	if (idx == 0){
		printf("gau = %f, %f\n",gau[0], gau[1]); // print gau
		printf("nh1,2,3,4,5 = %d,%d,%d,%d,%d\n", nh1,nh2,nh3,nh4,nh5); //print nh
		
		for (int i =0; i <4; i ++){//print 4 momentum
			printf("p1[%d]: %f\n", i, p1[i]);
			printf("p2[%d]: %f\n", i, p2[i]);
			printf("p3[%d]: %f\n", i, p3[i]);
			printf("p4[%d]: %f\n", i, p4[i]);
			printf("p5[%d]: %f\n", i, p5[i]);
		}
		
		for (int i = 0; (idx == 0) && (i < 7); i++){ // print wave functions
			printf("w01[%d]: %f + %fi\n", i, w01[i].re, w01[i].im);
			printf("w02[%d]: %f + %fi\n", i, w02[i].re, w02[i].im);
			printf("w03[%d]: %f + %fi\n", i, w03[i].re, w03[i].im);
			printf("w04[%d]: %f + %fi\n", i, w04[i].re, w04[i].im);
			printf("w05[%d]: %f + %fi\n", i, w05[i].re, w05[i].im);
			printf("w06[%d]: %f + %fi\n", i, w06[i].re, w06[i].im);
			printf("w07[%d]: %f + %fi\n", i, w07[i].re, w07[i].im);
			printf("w08[%d]: %f + %fi\n", i, w08[i].re, w08[i].im);
		}

	}*/
	
	fvoxx0(w02,w03,gau,w06);
	fvoxx0(w06,w04,gau,w07);
	iovxxx(w01,w07,w05,gau,&amp); 
	ampsum = ADD(ampsum, amp);
	
	
	fvixx0(w01,w04,gau,w07);
	fvoxx0(w02,w05,gau,w08);
	iovxxx(w07,w08,w03,gau,&amp);
	ampsum = ADD(ampsum, amp);
	
	
	fvoxx0(w02,w03,gau,w06);
	fvixx0(w01,w04,gau,w07);
	iovxxx(w07,w06,w05,gau,&amp);
	ampsum = ADD(ampsum, amp);
	
	
	fvoxx0(w02,w04,gau,w06);
	fvixx0(w01,w05,gau,w07);
	iovxxx(w07,w06,w03,gau,&amp);
	ampsum = ADD(ampsum, amp);
	
	
	fvixx0(w01,w03,gau,w07);
	fvixx0(w07,w04,gau,w08);
	iovxxx(w08,w02,w05,gau,&amp);
	ampsum = ADD(ampsum, amp);
	
	
	fvixx0(w01,w03,gau,w07);
	fvoxx0(w02,w04,gau,w06);
	iovxxx(w07,w06,w05,gau,&amp);
	ampsum = ADD(ampsum, amp);
	
	Amp_d[idx] = ampsum;

	
}
