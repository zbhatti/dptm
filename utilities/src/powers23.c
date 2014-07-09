#include <stddef.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/un.h>
#include <math.h>
#include <stdio.h>

int compareFunction(const void *a,const void *b) {
	int *x = (int *) a;
	int *y = (int *) b;
return *x - *y;
}

void populateSizes(int A[], int MaxVectors){
	int i, j, k, a;
	i = 0; k = 0; a = 0;
	
	int Power2 = 0;
	int Power3 = 0;
	int Power5 = 0;
	for(j=Power3; pow(5,i)*pow(3,j)*pow(2,k) < MaxVectors; j++){
		for(k=Power2; pow(5,i)*pow(3,j)*pow(2,k) < MaxVectors; k++){
			A[a++] = pow(5,i)*pow(3,j)*pow(2,k);
		}
		k=Power2;
	}
	qsort(A, a, sizeof(int), compareFunction);
}



int main(){
	int VectorSizes[1000];
	int MaxVectors = 400;
	populateSizes(VectorSizes, MaxVectors);
	for (int i=0; VectorSizes[i]<= MaxVectors; i++){
		printf("%d\n", VectorSizes[i]);
	}
}