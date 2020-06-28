#include <iostream>
#include <cstring>
using namespace std;
int num = 0;
int res_x = 0;
int res_y = 0;
char board[4][4] = {};

bool check_finish(int x,int y ){
	int p1 = 0;
	int p2 = 0;
	for(int j = 0; j < 4; j++){ //check row
		if (board[x][j] == 'x')
			p1++;
		else if (board[x][j] == 'o')
			p2++;
	}
	if(p1 == 4 || p2 == 4)
		return true;
	
	p1 = p2 = 0;
	for (int i = 0; i < 4; i++){ //check column
		if (board[i][y] == 'x')
			p1++;
		else if (board[i][y] == 'o')
			p2++;
	}
	if (p1 == 4 || p2 == 4)
		return true;
		
	p1 = p2 = 0;
	for (int i = 0; i < 4; i++){ //check diagonal from top left to right bottom
		if(board[i][i] == 'x')
			p1++;
		else if(board[i][i] == 'o')
			p2++;
	}
	if (p1 == 4 || p2 == 4)
		return true;
		
	p1 = p2 = 0;
	for (int i = 0; i < 4; i++){ //check diagonal from left bottom to top right
		if (board[i][3-i] == 'x')
			p1++;
		else if (board[i][3-i] == 'o')
			p2++;
	}
	if (p1 == 4 || p2 == 4)
		return true;
	return false;
}
int maxSearch(int,int);
int minSearch(int x, int y){ //Player two's turn;
	if (check_finish(x,y))
		return 1;
	if (num == 16)
		return 0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			if (board[i][j] == '.'){
				board[i][j] = 'o';
				num++;
				int tmp = maxSearch(i,j);
				board[i][j] = '.';
				num--;
				if (tmp == -1 || tmp == 0)
					return -1;				
			}
		}
	}
	return 1;
}
int maxSearch(int x, int y){ //Player one's turn
	if (check_finish(x,y))
		return -1;
	if (num == 16)
		return 0;
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			if(board[i][j] == '.'){
				board[i][j] = 'x';
				num++;
				int tmp = minSearch(i,j);
				board[i][j] = '.';
				num--;
				if (tmp == 1)
					return 1;
			}
		}
	}
	return -1;
}

bool solve(){
	for (int i = 0; i < 4; i++){
		for (int j = 0; j < 4; j++){
			if(board[i][j] == '.'){
				board[i][j] = 'x';
				num++;
				int tmp = minSearch(i,j);
				board[i][j] = '.';
				num--;
				if (tmp == 1){
					res_x = i;
					res_y = j;
					return true;
				}
			}
		}
	}
	return false;
}

int main(){
	char a;
	while(cin >> a){
		if(a == '$')
			break;
		num = 0;
		res_x = 0;
		res_y = 0;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				cin >> board[i][j];
				if(board[i][j] != '.')
					num++;
			}
		}
		if (solve()){
			cout << "(" << res_x << "," << res_y << ")" << endl;
		}
		else
			cout << "#####" << endl;
	}
	
}
