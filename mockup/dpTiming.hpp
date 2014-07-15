#include <vector>

class dpTiming{
	public:
		//kernel name
		std::vector<float> data; //pair strings with data points
		int localX;
		int localY;
		int localZ;
		float memoryCopyOut;
		float plan;
		float execute;
		float memoryCopyIn;
		float cleanUp;
};