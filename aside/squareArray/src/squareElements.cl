__kernel void squareElements(__global const float * Ain_d, __global float * Aout_d, const int N){
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	int idz = get_global_id(2);
	int Index = idx + idy*get_global_size(0) + idz*get_global_size(1)*get_global_size(2);
	//2D:int Index = idx + idy*get_global_size(0);
	
	if (Index<N){
		Aout_d[Index] = Ain_d[Index] * Ain_d[Index];
	}
}
