#include <iostream>
#include <queue>
using namespace std;

int m, n;
bool closed[5000][5000][2] = {};

struct Stat {
    int lm; // M at left side
    int lc; // C at left side
    bool isLeft; // position of boat
    int step; // Step taken
    Stat(int m, bool b = true, int s = 0): lm(m), lc(m), isLeft(b), step(s) {}
    bool validMove(int moveM, int moveC) {
    	if (moveM >= moveC || moveM == 0) //Condition in boat
    	{
	        if (isLeft){
	        	return (lm >= moveM && lc >= moveC && lm - moveM >= lc - moveC || lm == moveM)  //Condition at left side
					&& (m - lm + moveM >= m -lc + moveC || m - lm + moveM == 0)  //Condition at right side
					&& (!closed[lm - moveM][lc - moveC][!isLeft]); 
			}
			else{
				return (lm + moveM >= 0 && lc + moveC >= 0 && lm + moveM >= lc + moveC || lm + moveM == 0) //Condition at left side
					&& (m - lm -moveM >= m - lc - moveC || m - lm - moveM == 0) //Condition at right side
					&& (!closed[lm + moveM][lc + moveC][!isLeft]);
			}
		}
		else
			return false;
    }	
    Stat gen(int moveM, int moveC) {
    	Stat res = *this;
    	if (isLeft){
    		res.lm -= moveM;
    		res.lc -= moveC;
		}
		else{
			res.lm += moveM;
			res.lc += moveC;
		}
		res.isLeft = !res.isLeft;
		res.step++;
		closed[res.lm][res.lc][res.isLeft] = true; 
		return res; 

    }
};

queue<Stat> opened;

int main() {
    cin >> m >> n;
    opened.push(Stat(m));
    closed[m][m][1] = true;
    while (!opened.empty()) {
        Stat node = opened.front();
        opened.pop();
        if (node.lc == 0 && node.lm == 0) {
            cout << node.step << endl;
            return 0;
        }
        if (node.isLeft) { //Boat at left side
            for (int i = 0; i <= n; i++) {
                for (int j = 0; j <= n - i; j++) {
                	if (i == 0 && j == 0)continue;
                    if (node.validMove(i, j)) {
                        opened.push(node.gen(i, j));
                    }
                }
            }
        } 
		else { //Boat at right side
            for (int i = 0; i <= n; i++) {
                for (int j = 0; j <= n - i; j++) {
                	if (i == 0 && j == 0)continue;
                    if (node.validMove(i, j)) {
                        opened.push(node.gen(i, j));
                    }
                }
            }
        }
    }
	cout << "-1" << endl;
}
