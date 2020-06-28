#include <iostream>
#include <cstring>
using namespace std;
int chess[8][8] = {0};
int column[8] = {0};
int da[2*8] = {0}; //da is diagonal from left-top to right-bottom
int db[2*8] = {0}; //db is diagonal from left-bottom to right-top

void printChess(){
	for (int i = 0; i < 8; i++){
		for (int j = 0; j < 8; j++){
			if(column[j] == i+1)cout << j+ 1 << (i == 7?"\n":" ");
		}
	}
}

bool searchAns(int x){
	if(x == 8){
		printChess();
		return true;
	}
	bool flag = false;
	for (int j = 0; j < 8; j++){
		if(!column[j]&& !da[x-j+8] && !db[x+j]){
			chess[x][j] = 1;
			column[j] = x+1;
			da[x-j+8] = 1;
			db[x+j] = 1;
			flag = searchAns(x+1);
			if (!flag){
				chess[x][j] = 0;
				column[j] = 0;
				da[x-j+8] = 0;
				db[x+j] = 0;
			}
			else{
				break;
			}
		}
	}
	return false;
}

int main(){
	int n1, n2;
	cin >> n1 >> n2;
	chess[0][n1-1] = chess[1][n2-1] = 1;
	column[n1-1] = 1;
	column[n2-1] = 2;
	da[9-n1] = 1; 
	db[n1-1] = 1;
	da[10-n2] = 1;
	db[n2] = 1; 
	searchAns(2);
	return 0;	
}
