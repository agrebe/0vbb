#include "color_tensor.h"

int color_idx_1 [36 * 7];
int color_idx_2 [1296 * 13];

void color_tensor_1body(int * idx)
{
  int eps_sign1, eps_sign2, sign, ii(0);

  for(int i=0; i<3; ++i){
  for(int j=0; j<3; ++j){
  for(int k=0; k<3; ++k){
  for(int ip=0; ip<3; ++ip){
  for(int jp=0; jp<3; ++jp){
  for(int kp=0; kp<3; ++kp){

    if((i == j) || (i == k) || (j == k)){ continue; }
    if((ip == jp) || (ip == kp) || (jp == kp)){ continue; }

    eps_sign1 = ((i+1)%3 == j) ? 1 : -1; 
    eps_sign2 = ((ip+1)%3 == jp) ? 1 : -1; 
    sign      = eps_sign1 * eps_sign2;

    idx[7*ii+0]  = i;
    idx[7*ii+1]  = j;
    idx[7*ii+2]  = k;
    idx[7*ii+3]  = ip; 
    idx[7*ii+4]  = jp; 
    idx[7*ii+5]  = kp; 
    idx[7*ii+6] = sign;
    ii++;

  }}}}}}
}

void color_tensor_2bodies(int * idx)
{
  int eps_sign1, eps_sign2, eps_sign3, eps_sign4, sign, ii(0);

  for(int i=0; i<3; ++i){
  for(int j=0; j<3; ++j){
  for(int k=0; k<3; ++k){
  for(int l=0; l<3; ++l){
  for(int m=0; m<3; ++m){
  for(int n=0; n<3; ++n){
  for(int ip=0; ip<3; ++ip){
  for(int jp=0; jp<3; ++jp){
  for(int kp=0; kp<3; ++kp){
  for(int lp=0; lp<3; ++lp){
  for(int mp=0; mp<3; ++mp){
  for(int np=0; np<3; ++np){

    if((i == j) || (i == k) || (j == k)){ continue; }
    if((l == m) || (l == n) || (m == n)){ continue; }
    if((ip == jp) || (ip == kp) || (jp == kp)){ continue; }
    if((lp == mp) || (lp == np) || (mp == np)){ continue; }

    eps_sign1 = ((i+1)%3 == j) ? 1 : -1;
    eps_sign2 = ((l+1)%3 == m) ? 1 : -1;
    eps_sign3 = ((ip+1)%3 == jp) ? 1 : -1;
    eps_sign4 = ((lp+1)%3 == mp) ? 1 : -1;
    sign      = eps_sign1 * eps_sign2 * eps_sign3 * eps_sign4;

    idx[13*ii+0]  = i;
    idx[13*ii+1]  = j;
    idx[13*ii+2]  = k;
    idx[13*ii+3]  = l;
    idx[13*ii+4]  = m;
    idx[13*ii+5]  = n;
    idx[13*ii+6]  = ip;
    idx[13*ii+7]  = jp;
    idx[13*ii+8]  = kp;
    idx[13*ii+9]  = lp;
    idx[13*ii+10] = mp;
    idx[13*ii+11] = np;
    idx[13*ii+12] = sign;
    ii++;

  }}}}}}}}}}}}
}

