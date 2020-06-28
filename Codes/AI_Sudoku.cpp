#include <iostream>
#include <cstring>
using namespace std;
bool row[10][10];
bool column[10][10];
bool square[10][10];
int chess[10][10];

bool searchAns(int i, int j){
	if (i == 10)
		return true;
	int l = 3 *((i - 1)/ 3) + (j - 1)/3 + 1;
	bool flag = false;
	if (chess[i][j] == 0){
		for (int k = 1; k <= 9; k++){
			if (!row[i][k] && !column[j][k] && !square[l][k]){
				row[i][k] = true;
				column[j][k] = true;
				square[l][k] = true;
				chess[i][j] = k;  
				if (j == 9)
					flag = searchAns(i+1,1);
				else
					flag = searchAns(i,j+1);
				if(!flag){
					chess[i][j] = 0;
					row[i][k] = false;
					column[j][k] = false;
					square[l][k] = false;
				}
				else
					return true;
			}
		}
	}
	else{
		if (j == 9)
			return searchAns(i+1,1);
		else
			return searchAns(i,j+1);
	}
	return false;
}

int main(){
	int n;
	cin >> n;
	while (n--){
		memset(chess,0,sizeof(chess));
		memset(square,0,sizeof(square));
		memset(column,0,sizeof(column));
		memset(row,0,sizeof(row));
		char s;
		for (int i = 1; i <= 9; i++){
			for (int j = 1; j <= 9; j++){
				cin >> s;
				chess[i][j] = s - '0';
				int k = 3 *((i - 1)/ 3) + (j - 1)/3 + 1;
				if(chess[i][j]){
					square[k][chess[i][j]]= true;
					column[j][chess[i][j]]= true;
					row[i][chess[i][j]] = true; 	
				}
			}
		}
		searchAns(1,1);
		for (int i= 1; i <= 9; i++){
			for (int j = 1; j <= 9 ; j++){
				cout << chess[i][j];
			}
			cout << endl;
		}
	}
	return 0;
}
