
__kernel void Oxxxxx(__global const float * Ain_d, __global float * Aout_d, const int N){
	int idx = get_global_id(0);
	if (idx<N)
		Aout_d[idx]=Ain_d[idx]*Ain_d[idx];
	
}
