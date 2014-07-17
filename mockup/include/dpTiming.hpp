#include <vector>
#include <string>

class dpTiming{
	public:
		std::string name;//kernel name
		std::vector<float> data; //pair strings with data points, kernel must set this in the init() function
		size_t* localSize;
		float memoryCopyOut;
		float plan;
		float execute;
		float memoryCopyIn;
		float cleanUp;
		
		//add print method
};