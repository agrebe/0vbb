WeylMat T_exp_0[729];
WeylMat T_exp_1[729];
WeylMat T_exp_2[729];
WeylMat T_exp_3[729];
WeylMat T_exp_4[729];
WeylMat T_exp_5[729];
WeylMat T_exp_6[729];
WeylMat T_exp_7[729];
WeylMat T_exp_8[729];
WeylMat T_exp_9[729];
WeylMat T_exp_10[729];
for (int a = 0; a < 3; a ++) {
for (int b = 0; b < 3; b ++) {
for (int c = 0; c < 3; c ++) {
for (int d = 0; d < 3; d ++) {
for (int e = 0; e < 3; e ++) {
for (int f = 0; f < 3; f ++) {
T_exp_0[243*a+81*b+27*c+9*d+3*e+f] = S_tmxp_CG[3*a+b] * S_tmp_T_CG[3*c+d] * S_tmxp_CG[3*e+f];
T_exp_1[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_T_CG[3*a+b] * S_tmp_CG[3*c+d] * S_tmp_T[3*e+f];
T_exp_2[243*a+81*b+27*c+9*d+3*e+f] = S_tmxp_CG[3*a+b] * S_tmp_T_CG[3*c+d] * S_tmp_CG[3*e+f];
T_exp_3[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_T_CG[3*a+b] * S_tmxp_CG[3*c+d] * S_tmp_T[3*e+f];
T_exp_4[243*a+81*b+27*c+9*d+3*e+f] = S_tmxp_T_CG[3*a+b] * S_tmp_CG[3*c+d] * S_tmp_T[3*e+f];
T_exp_5[243*a+81*b+27*c+9*d+3*e+f] = S_tmxp_CG[3*a+b] * S_tmxp_T_CG[3*c+d] * S_tmp_CG[3*e+f];
T_exp_6[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_T_CG[3*a+b] * S_tmp_CG[3*c+d] * S_tmxp_T[3*e+f];
T_exp_7[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_CG[3*a+b] * S_tmp_T_CG[3*c+d] * S_tmp_CG[3*e+f];
T_exp_8[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_T_CG[3*a+b] * S_tmxp_CG[3*c+d] * S_tmp_T_CG[3*e+f];
T_exp_9[243*a+81*b+27*c+9*d+3*e+f] = S_tmxp_T_CG[3*a+b] * S_tmp_CG[3*c+d] * S_tmp_T_CG[3*e+f];
T_exp_10[243*a+81*b+27*c+9*d+3*e+f] = S_tmp_T_CG[3*a+b] * S_tmp_CG[3*c+d] * S_tmp_T_CG[3*e+f];
}
}
}
}
}
}
WeylMat P_exp_0[81];
WeylMat P_exp_1[81];
WeylMat P_exp_2[81];
WeylMat P_exp_3[81];
for (int a = 0; a < 3; a ++) {
for (int b = 0; b < 3; b ++) {
for (int c = 0; c < 3; c ++) {
for (int d = 0; d < 3; d ++) {
P_exp_0[27*a+9*b+3*c+d] = S_tmxp_CG[3*a+b] * S_tmp_T_CG[3*c+d];
P_exp_1[27*a+9*b+3*c+d] = S_tmxp_CG[3*a+b] * S_tmp_T[3*c+d];
P_exp_2[27*a+9*b+3*c+d] = S_tmp_CG[3*a+b] * S_tmp_T_CG[3*c+d];
P_exp_3[27*a+9*b+3*c+d] = S_tmxp_CG[3*a+b] * S_tmxp_T[3*c+d];
}
}
}
}
