#include <iostream>
#include <random>

using namespace std;

int main() {

	std::mt19937 generator (123);
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	double x = dis(generator);

	cout << x << endl;

	return 0;
}