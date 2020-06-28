#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;
ofstream fout ("output.txt");
double object[6][6] = {
    {0},
    {0,0,1,1,0,0},
    {0,0,1,1,0,0},
    {0,0,2,2,0,0},
    {0,1,1,1,1,0},
    {0}
};
double core[3][3] = {
    {0,0,0},{1,0,-1},{1,0,-1}
};
double real_image[4][4] = {
    {-2,-2,2,2},
    {-4,-4,4,4},
    {-4,-3,3,4},
    {-3,-2,2,3}
};
double image[4][4];
double partialK[3][3];
double calculateImage(int m, int n){
    double sum = 0;
    for (int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            sum += core[i][j]*object[m+i][n+j];
        }
    }
    return sum;
}
double calculatePartial(int m, int n){
    double sum = 0;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            sum  += (image[i][j]-real_image[i][j]) * object[m+i][n+j];
        }
    }
    return sum/8;
}

int main(){
    fout<< "image :" << endl;
    for (int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            image[i][j] = calculateImage(i,j);
            fout<< image[i][j] << " ";
        }
        fout<< endl;
    }

    fout<< endl <<  "Lost function: " ; 
    double sum = 0;
    for (int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            sum+=  pow(image[i][j]-real_image[i][j],2);
        }
    }
    sum /= 16;
    fout<< sum << endl << endl;
    fout<< "y - Y:" << endl;
    for (int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            fout<< image[i][j]-real_image[i][j] << " ";
        }
        fout<< endl;
    }
    fout<< endl << "partial K:" << endl;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            partialK[i][j] =  calculatePartial(i,j) ;
            fout<< partialK[i][j] << " ";
        }
        fout<< endl;
    }
    fout<< endl << "new core:" << endl;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            core[i][j] =  core[i][j] - 0.01 * (partialK[i][j]) ;
            fout<< core[i][j] << " ";
        }
        fout<< endl;
    }
    fout<< endl << "new image :" << endl;
    for (int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            image[i][j] = calculateImage(i,j);
            fout<< image[i][j] << " ";
        }
        fout<< endl;
    }

    fout<< endl <<  "Lost function: " ; 
    sum = 0;
    for (int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            sum+=  pow(image[i][j]-real_image[i][j],2);
        }
    }
    sum /= 16;
    fout<< sum << endl << endl;
}