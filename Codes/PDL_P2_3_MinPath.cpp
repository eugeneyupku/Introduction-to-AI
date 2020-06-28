#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
int a[101][101] = {0};
int n;

void floyd(){
	for (int k = 0; k < n; k++){
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				if(i == j || i == k || j == k)continue;
				if(a[i][j] > a[i][k] + a[k][j]){
					a[i][j] = a[i][k] + a[k][j];
				}
			}
		}		
	}

}

int main(){
	cin >> n;
	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++){
			cin >> a[i][j];
		}
	}
	floyd();
	cout << a[0][n-1] << endl;
}
