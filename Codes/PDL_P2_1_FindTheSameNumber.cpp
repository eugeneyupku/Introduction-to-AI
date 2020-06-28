#include <iostream>
using namespace std;
int main(){
	int n;
	cin >> n;
	int num[1000] = {0};
	for (int i = 0; i < n; i++){
		cin >> num[i];
		for (int j = 0; j < i; j++){
			if (num[j] == num[i])cout << num[i] << endl;
		}
	}
	return 0;
} 
