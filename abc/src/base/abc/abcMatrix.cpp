/*************************************************************************
	> File Name: abcMatrix.cpp
	> Author: Cunxi Yu 
	> Mail:  cunxi.yu@cornell.edu
	> Created Time: Tue Dec  5 14:03:10 2017
 ************************************************************************/


#include "abc.h"
#include "base/abc/abc.h"
#include "base/main/main.h"
#include "map/mio/mio.h"
#include "bool/dec/dec.h"
#include "opt/fxu/fxu.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdlib.h> 
#include <fstream>
#ifdef ABC_USE_CUDD
#include "bdd/extrab/extraBdd.h"
#endif

ABC_NAMESPACE_IMPL_START



using namespace std;

static vector<int> collectId(Abc_Ntk_t * pNtk);

void printMaxSum_NbyN_fOut(vector<vector<int> > matrix_vec, int ROW, int N, int k, char * fOut, int visual)
{
/*
    //int mat[ROW][N];
    ofstream fout; fout.open(fOut);
    int ** mat;
    mat = (int**)malloc(sizeof(int*)*ROW);
    for (int i = 0; i < ROW; i++)
        mat[i] = (int*)malloc(sizeof(int)*N);
    
    for(int i=0; i<ROW;i++)
        for(int j=0; j<N;j++)
            mat[i][j] = matrix_vec[i][j];
            //cout<< matrix_vec[i][j];
    // Print the result matrix
    if(k==-1){
    if(visual == 1){
        for(int i=0; i<ROW;i++)
        {
             for(int j=0; j<N;j++){
                fout<< matrix_vec[i][j];
                if(j<N-1)
                    fout<<",";
             }
                fout << endl;
        }
        return ;
    }
    else{
         for(int i=0; i<ROW;i++)
        {
             for(int j=0; j<N;j++){
                fout<< matrix_vec[i][j];
                if( !(j==N-1 & i==ROW-1))
                    fout<<",";
             }
        }
    }
    }
    // k must be smaller than or equal to n
    if (k > N) return;
 
    // 1: PREPROCESSING
    // To store sums of all strips of size k x 1
    //int stripSum[ROW-k+1][N];
    int ** stripSum;
    stripSum = (int**)malloc(sizeof(int*)*(ROW-k+1));
    for (int i = 0; i < ROW-k+1; i++)
        stripSum[i] =  (int*)malloc(sizeof(int)*N);
    
    // Go column by column
    for (int j=0; j<N; j++)
    {
        // Calculate sum of first k x 1 rectangle
        // in this column
        int sum = 0;
        for (int i=0; i<k; i++)
            //sum += (mat[i][j]);
            sum += abs(mat[i][j]);
        stripSum[0][j] = sum;
 
        // Calculate sum of remaining rectangles
        for (int i=1; i<ROW-k+1; i++)
        {
            //sum += (mat[i+k-1][j] - (mat[i-1][j]);
            sum += (abs(mat[i+k-1][j]) - abs(mat[i-1][j]));
            stripSum[i][j] = sum;
        }
    }
    for(int i=0;i<ROW-k;i++)
    {
        for(int j=0;j<N;j++)
            cout<<stripSum[i][j]<<" ";
        cout<<endl;
    }
    
    cout<<endl;
    // max_sum stores maximum sum and its
    // position in matrix
    int max_sum = INT_MIN, *pos = NULL;
    int pos1, pos2; 
    // 2: CALCULATE SUM of Sub-Squares using stripSum[][]
    for (int i=0; i<ROW-k+1; i++)
    {
        // Calculate and print sum of first subsquare
        // in this row
        int sum = 0;
        for (int j = 0; j<k; j++)
            sum += stripSum[i][j];
 
        // Update max_sum and position of result
        if (sum > max_sum)
        {
            max_sum = sum;
            pos = &(mat[i][0]);
            pos1 = i;
            pos2 = 0;
        }
 
        // Calculate sum of remaining squares in
        // current row by removing the leftmost
        // strip of previous sub-square and adding
        // a new strip
        for (int j=1; j<N-k+1; j++)
        {
            sum += (stripSum[i][j+k-1] - stripSum[i][j-1]);
 
            // Update max_sum and position of result
            if (sum > max_sum)
            {
                max_sum = sum;
                pos = &(mat[i][j]);
                pos1 = i;
                pos2 = j;
            }
        }
    }
    // Print the result matrix
    if(visual==1){
        for (int i=0; i<k; i++)
        {
            for (int j=0; j<k; j++){
                //cout << *(pos + i*N + j) << " ";
                fout<<mat[pos1+i][pos2+j];
                if(j<k-1)
                    fout<<",";
            }
            fout << endl;
        }
    }
    else{
        for (int i=0; i<k; i++)
        {
            for (int j=0; j<k; j++){
                //cout << *(pos + i*N + j) << " ";
                fout<<mat[pos1+i][pos2+j];
                if( !(j==k-1 & i==k-1))
                    fout<<",";
            }
            if(visual == 1)
            fout << endl;
        }
    }
    */
    return ;
}


void printMaxSum_NbyN(vector<vector<int> > matrix_vec, int ROW, int N, int k)
{
    //int mat[ROW][N];
/*
        int ** mat;
    mat = (int**)malloc(sizeof(int*)*ROW);
    for (int i = 0; i < ROW; i++)
        mat[i] = (int*)malloc(sizeof(int)*N);
    
    for(int i=0; i<ROW;i++)
        for(int j=0; j<N;j++)
            mat[i][j] = matrix_vec[i][j];
            //cout<< matrix_vec[i][j];
    // Print the result matrix
    if(k==-1){
        for(int i=0; i<ROW;i++)
        {
            for(int j=0; j<N;j++){
                cout<< matrix_vec[i][j];
                if(j<N-1)
                    cout<<",";
            }
             cout << endl;
        }
        return ;
    }
    // k must be smaller than or equal to n
    if (k > N) return;
 
    // 1: PREPROCESSING
    // To store sums of all strips of size k x 1
    //int stripSum[ROW-k+1][N];
    int ** stripSum;
    stripSum = (int**)malloc(sizeof(int*)*(ROW-k+1));
    for (int i = 0; i < ROW-k+1; i++)
        stripSum[i] =  (int*)malloc(sizeof(int)*N);
    
    // Go column by column
    for (int j=0; j<N; j++)
    {
        // Calculate sum of first k x 1 rectangle
        // in this column
        int sum = 0;
        for (int i=0; i<k; i++)
            //sum += (mat[i][j]);
            sum += abs(mat[i][j]);
        stripSum[0][j] = sum;
 
        // Calculate sum of remaining rectangles
        for (int i=1; i<ROW-k+1; i++)
        {
            //sum += (mat[i+k-1][j] - (mat[i-1][j]);
            sum += (abs(mat[i+k-1][j]) - abs(mat[i-1][j]));
            stripSum[i][j] = sum;
        }
    }
   /* 
    for(int i=0;i<ROW-k;i++)
    {
        for(int j=0;j<N;j++)
            cout<<stripSum[i][j]<<" ";
        cout<<endl;
    }
    
    cout<<endl;
    // max_sum stores maximum sum and its
    // position in matrix
    int max_sum = INT_MIN, *pos = NULL;
    int pos1, pos2; 
    // 2: CALCULATE SUM of Sub-Squares using stripSum[][]
    for (int i=0; i<ROW-k+1; i++)
    {
        // Calculate and print sum of first subsquare
        // in this row
        int sum = 0;
        for (int j = 0; j<k; j++)
            sum += stripSum[i][j];
 
        // Update max_sum and position of result
        if (sum > max_sum)
        {
            max_sum = sum;
            pos = &(mat[i][0]);
            pos1 = i;
            pos2 = 0;
        }
 
        // Calculate sum of remaining squares in
        // current row by removing the leftmost
        // strip of previous sub-square and adding
        // a new strip
        for (int j=1; j<N-k+1; j++)
        {
            sum += (stripSum[i][j+k-1] - stripSum[i][j-1]);
 
            // Update max_sum and position of result
            if (sum > max_sum)
            {
                max_sum = sum;
                pos = &(mat[i][j]);
                pos1 = i;
                pos2 = j;
            }
        }
    }
    // Print the result matrix
    for (int i=0; i<k; i++)
    {
        for (int j=0; j<k; j++){
            cout<<mat[pos1+i][pos2+j];
            if(j<k-1)
                cout<<",";
        }
        cout << endl;
    }
*/
    return ;
}


int kadane(int* arr, int* start, int* finish, int n)
{
    // initialize sum, maxSum and
    int sum = 0, maxSum = 0, i;
 
    // Just some initial value to check for all negative values case
    *finish = -1;
 
    // local variable
    int local_start = 0;
 
    for (i = 0; i < n; ++i)
    {
        sum += arr[i];
        if (sum < 0)
        {
            sum = 0;
            local_start = i+1;
        }
        else if (sum > maxSum)
        {
            maxSum = sum;
            *start = local_start;
            *finish = i;
        }
    }
 
     // There is at-least one non-negative number
    if (*finish != -1)
        return maxSum;
 
    // Special Case: When all numbers in arr[] are negative
    maxSum = arr[0];
    *start = *finish = 0;
 
    // Find the maximum element in array
    for (i = 1; i < n; i++)
    {
        if (arr[i] > maxSum)
        {
            maxSum = arr[i];
            *start = *finish = i;
        }
    }
    return maxSum;
}



// The main function that finds maximum sum rectangle in M[][]
//void findMaxSum(vector<vector<int> > M, int ROW, int COL)
void findMaxSum(vector<vector<int> > M, int ROW, int COL)
{
    // Variables to store the final output
    int maxSum = 0, finalLeft, finalRight, finalTop, finalBottom;
 
    int left, right, i;
    int temp[ROW], sum, start, finish;
 
    // Set the left column
    for (left = 0; left < COL; ++left)
    {
        // Initialize all elements of temp as 0
        memset(temp, 0, sizeof(temp));
 
        // Set the right column for the left column set by outer loop
        for (right = left; right < COL; ++right)
        {
           // Calculate sum between current left and right for every row 'i'
            for (i = 0; i < ROW; ++i)
                temp[i] += M[i][right];
 
            // Find the maximum sum subarray in temp[]. The kadane() 
            // function also sets values of start and finish.  So 'sum' is 
            // sum of rectangle between (start, left) and (finish, right) 
            //  which is the maximum sum with boundary columns strictly as
            //  left and right.
            sum = kadane(temp, &start, &finish, ROW);
 
            // Compare sum with maximum sum so far. If sum is more, then 
            // update maxSum and other output values
            if (sum > maxSum)
            {
                maxSum = sum;
                finalLeft = left;
                finalRight = right;
                finalTop = start;
                finalBottom = finish;
            }
        }
    }
 
    // Print final values
    printf("(Top, Left) (%d, %d)n", finalTop, finalLeft);
    printf("(Bottom, Right) (%d, %d)n", finalBottom, finalRight);
    printf("Max sum is: %dn", maxSum);
}



vector<int> collectId(Abc_Ntk_t * pNtk)
{
    vector<int> vec_id;
    int i; Abc_Obj_t * pObj;
    Abc_NtkForEachObj(pNtk,pObj,i){
        vec_id.push_back(pObj->Id);
    }
    return vec_id;
}



int convert2binaryVec(int x){
    if(x==1)
        return -1;
    else
        return 1;
}

void cpp_test(Abc_Ntk_t * pNtkOld, int k)
{
    //Abc_Ntk_t * pNtk = Abc_NtkStrash(pNtkOld, 0, 0, 0);
    Abc_Ntk_t * pNtk = pNtkOld;
    vector<int> vec_id; vec_id = collectId(pNtk);
    printf("Total Number of Nodes (including PI/PO nodes): %d\n", vec_id.size());
    int row = 0; row = vec_id.size()-1;
    unsigned long column = 0; 
    vector<vector<int> > m;
    vector<int> zeros;
    for(int i=0;i<vec_id.size(); i++){
        zeros.push_back(0);
    }
    for(int i=0;i<vec_id.size();i++){
        m.push_back(zeros);
    }
    // (i,j) => node_i > node_j
    int i=0;
    Abc_Obj_t * pObj;
    int ci_num=0;
    int co_num=0;
    Abc_NtkForEachCi(pNtk, pObj, i)
        ci_num++;
    Abc_NtkForEachCo(pNtk, pObj, i)
        co_num++;
    

    ci_num++;
    //cout<<ci_num<<" "<<co_num<<endl;
    Abc_Obj_t * pFanin0, * pFanin1;
    int i2;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(Abc_ObjFaninNum(pObj)==2){
            m[Abc_ObjId(Abc_ObjFanin0(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC0(pObj));
            m[Abc_ObjId(Abc_ObjFanin1(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC1(pObj));
        }
        if(Abc_ObjFaninNum(pObj)==1){
            m[Abc_ObjId(Abc_ObjFanin0(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC0(pObj));
        }
    }
    //printf("Matrix size is %lu-by-%lu(%lu-%lu)\n", m.size()-1, m.size()-ci_num, row, row-ci_num+1);
    vector<int> z;
    for(int i=0;i<row-ci_num+1;i++)
        z.push_back(0);
    vector<vector<int> > m_final;
    for(int i=0;i<row;i++)
        m_final.push_back(z);
    
//    z.clear();
    
    column = row - ci_num+1;
    //int f[row][column];
    for(int i=1;i<m.size();i++){
        for(int i1=ci_num;i1<m[i].size();i1++){
            //f[i-1][i1-ci_num] = m[i][i1];
            m_final[i-1][i1-ci_num] = m[i][i1];
     //       printf("%i ", m[i][i1]);
        }
     //   printf("\n");
    }
    m.clear(); z.clear();
    //cout<<m_final.size()<<" "<<m_final[0].size()<<endl;
    // m_final is the returned matrix
    /* print original matrix 
    for(int i=0;i<m_final.size();i++){
        for(int i1=0;i1<m_final[0].size();i1++){
            printf("%i ", m_final[i][i1]);
        }
        printf("\n");
    }
    
    */
    printf("Resulting size of matrix %lu:%lu\n", m_final.size(), m_final[0].size());
    assert(row = m_final.size());assert(column = m_final[0].size());
    printMaxSum_NbyN(m_final, row,column, k);m_final.clear();
    return;

}


void cpp_test2(Abc_Ntk_t * pNtkOld, int k, char * fOut, int visual)
{
    Abc_Ntk_t * pNtk = Abc_NtkStrash(pNtkOld, 0, 0, 0);
    vector<int> vec_id; vec_id = collectId(pNtk);
    printf("Total Number of Nodes (including PI/PO nodes): %d\n", vec_id.size());
    int row = 0; row = vec_id.size()-1;
    unsigned long column = 0; 
    vector<vector<int> > m;
    vector<int> zeros;
    for(int i=0;i<vec_id.size(); i++){
        zeros.push_back(0);
    }
    for(int i=0;i<vec_id.size();i++){
        m.push_back(zeros);
    }
    // (i,j) => node_i > node_j
    int i=0;
    Abc_Obj_t * pObj;
    int ci_num=0;
    int co_num=0;
    Abc_NtkForEachCi(pNtk, pObj, i)
        ci_num++;
    Abc_NtkForEachCo(pNtk, pObj, i)
        co_num++;
    

    ci_num++;
    //cout<<ci_num<<" "<<co_num<<endl;
    Abc_Obj_t * pFanin0, * pFanin1;
    int i2;
    Abc_NtkForEachObj(pNtk,pObj,i){
        //printf("Obj ID %i\n", pObj->Id);
        if(Abc_ObjFaninNum(pObj)==2){
            m[Abc_ObjId(Abc_ObjFanin0(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC0(pObj));
            m[Abc_ObjId(Abc_ObjFanin1(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC1(pObj));
        }
        if(Abc_ObjFaninNum(pObj)==1){
            m[Abc_ObjId(Abc_ObjFanin0(pObj))][Abc_ObjId(pObj)] = convert2binaryVec(Abc_ObjFaninC0(pObj));
        }
    }
    //printf("Matrix size is %lu-by-%lu(%lu-%lu)\n", m.size()-1, m.size()-ci_num, row, row-ci_num+1);
    vector<int> z;
    for(int i=0;i<row-ci_num+1;i++)
        z.push_back(0);
    vector<vector<int> > m_final;
    for(int i=0;i<row;i++)
        m_final.push_back(z);
    
//    z.clear();
    
    column = row - ci_num+1;
    //int f[row][column];
    for(int i=1;i<m.size();i++){
        for(int i1=ci_num;i1<m[i].size();i1++){
            //f[i-1][i1-ci_num] = m[i][i1];
            m_final[i-1][i1-ci_num] = m[i][i1];
     //       printf("%i ", m[i][i1]);
        }
     //   printf("\n");
    }
    m.clear(); z.clear();
    //cout<<m_final.size()<<" "<<m_final[0].size()<<endl;
    // m_final is the returned matrix
    /* print original matrix 
    for(int i=0;i<m_final.size();i++){
        for(int i1=0;i1<m_final[0].size();i1++){
            printf("%i ", m_final[i][i1]);
        }
        printf("\n");
    }
    
    */
    printf("Resulting size of matrix %lu:%lu\n", m_final.size(), m_final[0].size());
    assert(row = m_final.size());assert(column = m_final[0].size());
    printMaxSum_NbyN_fOut(m_final, row,column, k, fOut, visual);m_final.clear();
    return;

}

vector<int> sum_of_row(vector<vector<int> > matrix, int row, int column)
{
    vector<int> sum;
    int temp = 0;
    for(int i=0; i<row; i++)
    {
        temp = 0;
        for(int i2=0; i2<column; i2++)
        {
            temp+=matrix[i][i2];
        }
        sum.push_back(temp);
    }
    return sum;
}


vector<int> sum_of_column(vector<vector<int> > matrix, int row, int column)
{
    vector<int> sum;
    int temp = 0;
    for(int i=0; i<column; i++)
    {
        temp = 0;
        for(int i2=0; i2<row; i2++)
        {
            temp+=matrix[i2][i];
        }
        sum.push_back(temp);
    }
    return sum;
}

void vectorPrint(vector<int> v)
{
    for(int i=0; i<v.size();i++)
    {
        cout<<v[i];
        if(i<v.size()-1)
            cout<<",";
    }
    return ;
}


void stats_of_Matrix(vector<vector<int> > matrix)
{
    int row = matrix.size();
    assert(row > 0);
    int column = matrix[0].size();
    assert(column > 0);
    vector<int> sumRow, sumColumn;
    sumRow = sum_of_row(matrix, row, column);
    sumColumn = sum_of_column(matrix, row, column);
    cout<<"Sum of Rows:\n";
    vectorPrint(sumRow);
    cout<<endl;
    cout<<"Sum of Columns:\n";
    vectorPrint(sumColumn);
    cout<<endl;
    return;
}

void lut_matrix(Abc_Ntk_t * pNtk, int k, char * fOut, int visual)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    int max_id = *(vec_id.end()-1); 
    printf("Total Number of Nodes (including PI/PO nodes): %d;  max_id = %d\n", vec_id.size(), max_id);
    int row = 0; 
    row = max_id;
    //row = vec_id.size()-1;
    unsigned long column = 0; 
    vector<vector<int> > m;
    vector<int> zeros;
    for(int i=0;i<=row; i++){
        zeros.push_back(0);
    }
    for(int i=0;i<=row;i++){
        m.push_back(zeros);
    }
    // (i,j) => node_i > node_j
    int i=0;
    Abc_Obj_t * pObj;
    int ci_num=0;
    int co_num=0;
    Abc_NtkForEachCi(pNtk, pObj, i)
        ci_num++;
    Abc_NtkForEachCo(pNtk, pObj, i)
        co_num++;
    
    printf("Matrix size is %lu-by-%lu\n", m.size(), m[0].size());

    ci_num++;
    Abc_Obj_t * pFanin0, * pFanin1;
    int i2, fanin_i;
    int max_fanin=0; int max_fanout=0;
    
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        if(Abc_ObjFaninNum(pObj) > 0){
   //             printf("Obj %i\n", pObj->Id);
            Abc_ObjForEachFanin(pObj, pFanin0, fanin_i){
     //           printf("[loop] fanin id %i\n", pFanin0->Id);
                m[Abc_ObjId(pFanin0)][Abc_ObjId(pObj)] = 1;
            }
        }
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    stats_of_Matrix(m);
    //printf("Matrix size is %lu-by-%lu(%lu-%lu)\n", m.size()-1, m.size()-ci_num, row, row-ci_num+1);
    vector<int> z;
    for(int i=0;i<row-ci_num+1;i++)
        z.push_back(0);
    vector<vector<int> > m_final;
    for(int i=0;i<row;i++)
        m_final.push_back(z);
    
//    z.clear();
    
    column = row - ci_num+1;
    //int f[row][column];
    for(int i=1;i<m.size();i++){
        for(int i1=ci_num;i1<m[i].size();i1++){
            //f[i-1][i1-ci_num] = m[i][i1];
            m_final[i-1][i1-ci_num] = m[i][i1];
     //       printf("%i ", m[i][i1]);
        }
     //   printf("\n");
    }
    m.clear(); z.clear();
    //cout<<m_final.size()<<" "<<m_final[0].size()<<endl;
    // m_final is the returned matrix
    /* print original matrix 
    for(int i=0;i<m_final.size();i++){
        for(int i1=0;i1<m_final[0].size();i1++){
            printf("%i ", m_final[i][i1]);
        }
        printf("\n");
    }
    
    */
    printf("Resulting size of matrix %lu:%lu\n", m_final.size(), m_final[0].size());
    assert(row = m_final.size());assert(column = m_final[0].size());
    printMaxSum_NbyN_fOut(m_final, row, column, k, fOut, visual);
    stats_of_Matrix(m_final);
    m_final.clear();
    return;

}

void stats_of_Matrix_dump(vector<vector<int> > matrix, char * fOut)
{
    ofstream fout;
    fout.open(fOut);
    int row = matrix.size();
    assert(row > 0);
    int column = matrix[0].size();
    assert(column > 0);
    vector<int> sumRow, sumColumn;
    sumRow = sum_of_row(matrix, row, column);
    sumColumn = sum_of_column(matrix, row, column);
    //cout<<"Sum of Rows:\n";
    int row_size = sumRow.size();
    int col_size = sumColumn.size();
    for(int i=0;i<row_size;i++){
        fout<<sumRow[i];
        if(i<sumRow.size()-1)
            fout<<",";
    } 
    fout<<endl;
    for(int i=0;i<col_size;i++){
        fout<<sumColumn[i];
        if(i<sumColumn.size()-1)
            fout<<",";
    } 
    fout<<endl;
    return;
}

vector<vector<int> > matrixInit(int row, int column)
{
    vector<int> tmp;
    for(int i=0;i<column;i++)
        tmp.push_back(0);
    vector<vector<int> > m;
    for(int i=0;i<row;i++)
        m.push_back(tmp);
    return m;
}

void stats_of_Matrix_dump2(vector<vector<int> > matrix, char * fOut)
{
    ofstream fout;
    fout.open(fOut);
    int row = matrix.size();
    assert(row > 0);
    int column = matrix[0].size();
    assert(column > 0);
    vector<int> sumRow, sumColumn;
    sumRow = sum_of_row(matrix, row, column);
    sumColumn = sum_of_column(matrix, row, column);
    //cout<<"Sum of Rows:\n";
    int row_size = sumRow.size();
    int col_size = sumColumn.size();
    for(int i=0;i<row_size;i++){
        fout<<sumRow[i];
        if(i<sumRow.size()-1)
            fout<<",";
    } 
    fout<<endl;
    for(int i=0;i<col_size;i++){
        fout<<sumColumn[i];
        if(i<sumColumn.size()-1)
            fout<<",";
    } 
    fout<<endl;
    return;
}


void lut_matrix_featureSel2(Abc_Ntk_t * pNtk, char * fOut)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    int max_id = *(vec_id.end()-1); 
    //int max_id = * (max_element(vec_id.begin(), vec_id.end())); 
    printf("Total Number of Nodes (including PI/PO nodes): %d;  max_id = %d\n", vec_id.size(), max_id);
    Abc_Obj_t * pObj;
    int max_fanin=0; int max_fanout=0;
    int objNum=0;
    int i;
    Abc_NtkForEachObj(pNtk,pObj,i){
        objNum++;
    }
    vector<int> first, second;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        first.push_back(Abc_ObjFaninNum(pObj));
    }
    cout<<endl;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        second.push_back(Abc_ObjFanoutNum(pObj));
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    // each big element is a vector of all fanin IDs

    vector<vector<int> > fanin_matrix = matrixInit( objNum, max_fanin);
    vector<vector<int> > fanout_matrix = matrixInit( objNum, max_fanout);
    //cout<<fanin_matrix.size() <<", "<< i <<", "<< first.size() <<"," <<second.size()<<endl;
    assert(fanin_matrix.size() == (i-2) ); 
    assert((i-2) == first.size()); 
    assert(first.size() == second.size());
    assert(fanout_matrix.size() == i-2);
    cout<<fanin_matrix[0].size()<<","<<fanout_matrix[0].size()<<endl;
    int i2; Abc_Obj_t * pFanin, * pFanout;
    int incre1 = 0, incre2 = 0;
    Abc_NtkForEachObj(pNtk,pObj,i){
        incre2 = 0;
        Abc_ObjForEachFanin(pObj, pFanin, i2){
            fanin_matrix[incre1][incre2] = Abc_ObjId(pFanin);
            incre2++;
        }
        incre2 = 0;
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            fanout_matrix[incre1][incre2] = Abc_ObjId(pFanout);
            incre2++;
        }
        incre1++;
    }
    int column = first.size();
    for(int i=0;i<column;i++)
    {
        cout<<first[i];
        if(i<column-1)
            cout<<",";
    }
    cout<<endl;
    for(int i=0;i<column;i++)
    {
        cout<<second[i];
        if(i<column-1)
            cout<<",";
    }
    cout<<endl;
    for(int i=0;i<fanin_matrix[0].size();i++)
    {
        for(int i2 = 0; i2<column;i2++){
            cout<<fanin_matrix[i2][i];
            if(i2 < column-1)
                cout<<",";
        }
        cout<<endl;
    }
    for(int i=0;i<fanout_matrix[0].size();i++)
    {
        for(int i2 = 0; i2<column;i2++){
            cout<<fanout_matrix[i2][i];
            if(i2 < column-1)
                cout<<",";
        }
        cout<<endl;
    }


    return;

}


void network2edgelist(Abc_Ntk_t * pNtk, char * fOut)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    //#int max_id = *max_element(vec_id.begin(), vec_id.end()); 
    int max_id = *(vec_id.end()-1); 
    printf("Total Number of Nodes (including PI/PO nodes): %d;  max_id = %d\n", vec_id.size(), max_id);
    Abc_Obj_t * pObj;
    int max_fanin=0; int max_fanout=0;
    int objNum=0;
    int i;
    Abc_NtkForEachObj(pNtk,pObj,i){
        objNum++;
    }
    vector<int> first, second;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        first.push_back(Abc_ObjFaninNum(pObj));
    }
    cout<<endl;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        second.push_back(Abc_ObjFanoutNum(pObj));
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    ofstream fout;
    fout.open(fOut);
    int i2; Abc_Obj_t * pFanin, * pFanout;
    Abc_NtkForEachObj(pNtk,pObj,i){
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout)<<endl;
        }
    }

    return;

}

/*
int Abc_NtkGetCubeNum( Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pNode;
    int i, nCubes = 0;
    assert( Abc_NtkHasSop(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        if ( Abc_NodeIsConst(pNode) )
            continue;
        assert( pNode->pData );
        nCubes += Abc_SopGetCubeNum( (char *)pNode->pData );
    }
    return nCubes;
}

int Abc_NtkGetCubePairNum( Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pNode;
    int i;
    word nCubes, nCubePairs = 0;
    assert( Abc_NtkHasSop(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        if ( Abc_NodeIsConst(pNode) )
            continue;
        assert( pNode->pData );
        nCubes = (word)Abc_SopGetCubeNum( (char *)pNode->pData );
        if ( nCubes > 1 )
            nCubePairs += nCubes * (nCubes - 1) / 2;
    }
    return (int)(nCubePairs > (1<<30) ? (1<<30) : nCubePairs);
}

int Abc_NtkGetLitNum( Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pNode;
    int i, nLits = 0;
    assert( Abc_NtkHasSop(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        assert( pNode->pData );
        nLits += Abc_SopGetLitNum( (char *)pNode->pData );
    }
    return nLits;
}

int Abc_NtkGetLitFactNum( Abc_Ntk_t * pNtk )
{
    Dec_Graph_t * pFactor;
    Abc_Obj_t * pNode;
    int nNodes, i;
    assert( Abc_NtkHasSop(pNtk) );
    nNodes = 0;
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        if ( Abc_NodeIsConst(pNode) )
            continue;
        pFactor = Dec_Factor( (char *)pNode->pData );
        nNodes += 1 + Dec_GraphNodeNum(pFactor);
        Dec_GraphFree( pFactor );
    }
    return nNodes;
}


*/

void network2edgelist_comb(Abc_Ntk_t * pNtk, char * fOut)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    //#int max_id = *max_element(vec_id.begin(), vec_id.end()); 
    int max_id = *(vec_id.end()-1); 
    printf("Total Number of Nodes (including PI/PO nodes): %d;  max_id = %d\n", vec_id.size(), max_id);
    Abc_Obj_t * pObj;
    int max_fanin=0; int max_fanout=0;
    int objNum=0;
    int i;
    Abc_NtkForEachObj(pNtk,pObj,i){
        objNum++;
    }
    vector<int> first, second;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        first.push_back(Abc_ObjFaninNum(pObj));
    }
    cout<<endl;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        second.push_back(Abc_ObjFanoutNum(pObj));
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    ofstream fout;
    fout.open(fOut);
    int i2; Abc_Obj_t * pFanin, * pFanout;
    Abc_Obj_t * pNode;
 
    Abc_NtkForEachObj(pNtk,pObj,i){
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            fout<<Abc_ObjId(pObj)-1<<" "<<Abc_ObjId(pFanout)-1<<endl;
        }
    }

    return;

}

//sequntial AIG edgelist
void network2edgelist_seq(Abc_Ntk_t * pNtk, char * fOut)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    //#int max_id = *max_element(vec_id.begin(), vec_id.end()); 
    int max_id = *(vec_id.end()-1); 
    int nLatches = Abc_NtkLatchNum(pNtk);
    printf("Total Number of Nodes (including PI/PO nodes): %d; latches = %d; max_id = %d\n", 
                    vec_id.size(), nLatches, max_id);
    if(nLatches==0){
        printf("This is not a sequential circuit (no latch found).\n");
        //assert(0);
    }
    Abc_Obj_t * pObj;
    int max_fanin=0; int max_fanout=0;
    int objNum=0;
    int i;
    Abc_NtkForEachObj(pNtk,pObj,i){
        objNum++;
    }
    //vector<int> first, second;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        //first.push_back(Abc_ObjFaninNum(pObj));
    }
    cout<<endl;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        //second.push_back(Abc_ObjFanoutNum(pObj));
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    ofstream fout;
    fout.open(fOut);
    int i2; Abc_Obj_t * pFanin, * pFanout;
    /* with labels */
    /*
    Abc_NtkForEachObj(pNtk,pObj,i){
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            if(Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                // latch to X
                fout<<" 1"<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                // X to latch
                fout<<" 2"<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                // X to X
                fout<<" 3"<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                //latch to latch
                fout<<" 0"<<endl;
            }
            else{
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                fout<<" 10"<<endl;
                cout<<"ERROR \n";
                exit(0);
            }
        }
    }
    */
    Abc_NtkForEachObj(pNtk,pObj,i){
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            if(Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout)<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout)<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout)<<endl;
            }
            else if(!Abc_ObjIsLatch(pObj) && !Abc_ObjIsLatch(pFanout)){
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout)<<endl;
            }
            else{
                fout<<Abc_ObjId(pObj)<<" "<<Abc_ObjId(pFanout);
                fout<<" 10"<<endl;
                cout<<"ERROR \n";
                exit(0);
            }
        }
    }

    return;

}


void network2edgelist_structLabel(Abc_Ntk_t * pNtk, char * fOut, char *fOutclass, char *fOutfeat)
{
    vector<int> vec_id; vec_id = collectId(pNtk);
    //#int max_id = *max_element(vec_id.begin(), vec_id.end()); 
    int max_id = *(vec_id.end()-1); 
    int nLatches = Abc_NtkLatchNum(pNtk);
    printf("Total Number of Nodes (including PI/PO nodes): %d; latches = %d; max_id = %d\n", 
                    vec_id.size(), nLatches, max_id);
    if(nLatches==0){
        printf("This is not a sequential circuit (no latch found).\n");
        //assert(0);
    }
    Abc_Obj_t * pObj;
    int max_fanin=0; int max_fanout=0;
    int objNum=0;
    int i;
    Abc_NtkForEachObj(pNtk,pObj,i){
        objNum++;
    }
    //vector<int> first, second;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanin < Abc_ObjFaninNum(pObj))
            max_fanin = Abc_ObjFaninNum(pObj);
        //first.push_back(Abc_ObjFaninNum(pObj));
    }
    cout<<endl;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(max_fanout < Abc_ObjFanoutNum(pObj))
            max_fanout = Abc_ObjFanoutNum(pObj);
        //second.push_back(Abc_ObjFanoutNum(pObj));
    }
    cout<<"Max_fanin: "<<max_fanin<<"; Max_fanout: "<<max_fanout<<endl;
    ofstream fout,fout2,fout3;
    fout.open(fOut);
    fout2.open(fOutclass);
    fout3.open(fOutfeat);
    fout2<<"{";
    int i2; Abc_Obj_t * pFanin, * pFanout;
    Abc_Obj_t * f0, * f1;
    Abc_NtkForEachObj(pNtk,pObj,i){
        if(Abc_NodeIsMuxType(pObj)){
            fout2<<"\""<<Abc_ObjId(pObj)<<"\": "<<"[1]";
            //fout2<<"\""<<Abc_ObjId(pObj)<<"\": "<<"[0, 1]";
        }
        /*
        else if(Abc_NodeIsMuxControlType(pObj)){
            fout2<<"\""<<Abc_ObjId(pObj)<<"\": "<<"[1, 0]";}
            fout2<<"\""<<Abc_ObjId(pObj)<<"\": "<<"[1, 0]";}
            */
        else{
            fout2<<"\""<<Abc_ObjId(pObj)<<"\": "<<"[0]";
        }
        if(i<=max_id-1)
            fout2<<",";
        Abc_ObjForEachFanout(pObj, pFanout, i2){
            fout<<Abc_ObjId(pObj)-1<<" "<<Abc_ObjId(pFanout)-1<<endl;
        }
        if(Abc_ObjId(pObj)==0)
            continue;
        else if(Abc_ObjIsPo(pObj))
        {
              fout3<<"1,1,0"<<endl;
        }
        else if(Abc_ObjIsPi(pObj))
        {
              fout3<<"0,0,0"<<endl;
        }
        else if(Abc_ObjFaninNum(pObj)==2)
        {
                fout3<<Abc_ObjFaninC0(pObj)<<","<<Abc_ObjFaninC1(pObj)<<",1"<<endl;
        }
        else
        {
                f0 = Abc_ObjFanin0(pObj);
                fout3<<"-1,-1,0"<<endl;
        
        }


    }
    fout2<<"}";
    
    
    return;


}

















