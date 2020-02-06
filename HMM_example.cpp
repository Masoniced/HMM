#include "HMM.h"
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace std;

int main() {
	vector<int> a = { 0,2,1,2,1,2,3,0,1,2,0,0,0,1,2,1,0,1,3,2,1,0,0,1,0,1,2,3 };
	HMM solution(a, 3, 1000);
	solution.inference();
	cout << "Initial Distributon" << endl;
	copy(solution.Final_initial_prob.begin(), solution.Final_initial_prob.end(), ostream_iterator<double>(cout, ", "));
	cout << endl;
	cout << endl;
	cout << "State Transition Probability (A)" << endl;
	for_each(solution.Final_A_trans.begin(), solution.Final_A_trans.end(), [](vector<double>& row) \
	{copy(row.begin(), row.end(), ostream_iterator<double>(cout, ", ")); });
	cout << endl;
	cout << endl;
	cout << "Observation Transition Probability (B)" << endl;
	for_each(solution.Final_B_trans.begin(), solution.Final_B_trans.end(), [](vector<double>& row) \
	{copy(row.begin(), row.end(), ostream_iterator<double>(cout, ", ")); });
	cout << endl;

	system("pause");

	return 0;
}
