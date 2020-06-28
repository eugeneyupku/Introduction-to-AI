#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#define max_val 1e18;
using namespace std;
int dir[8][8] = {{0,1},{0,-1},{1,1},{1,-1},{1,0},{-1,1},{-1,-1},{-1,0}};

struct Point{
	int x;
	int y;
	Point(){}
	Point(int a, int b):x(a),y(b){}
	double operator-(const Point& a){
		return sqrt((x -a.x)*(x-a.x) + (y-a.y)*(y-a.y));
	}
};

Point p[2000];
int n;

double totalDis(Point a){
	double sum = 0;
	for (int i = 0; i < n; i++){
		sum += p[i] - a;
	}
	return sum;
}

bool out(int x, int y ){
	if (x < 0 || x > 10000 || y < 0 || y > 10000)return true;
	return false;
}

double hillClimb(){

	double curAns = max_val;
	Point tp;
	Point cur = p[0];
	for (double t = 1000; t > 1e-6; t*= 0.999){
		for (int i = 0; i < 8; ++i){
			tp.x = cur.x + dir[i][0]* t;
			tp.y = cur.y + dir[i][1]* t;
			double tmpAns = totalDis(tp);
			if(tmpAns < curAns){
				curAns = tmpAns;
				cur = tp;
				break;
			}
		}
	}
	return curAns;
}

int main(){
	while (cin >> n){
		for (int i = 0; i < n; i++){
			cin >> p[i].x >> p[i].y;
		}
		printf("%.0lf\n",hillClimb());
	}

}
