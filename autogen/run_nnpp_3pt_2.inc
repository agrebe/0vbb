tmp[0] -= sign * Trace(cgs * T_exp_0[243*a+81*j+27*c+9*g+3*b+k] * T_exp_1[243*e+81*l+27*f+9*h+3*d+i]);
tmp[1] += sign * Trace(cgs * S_tmxp_CG[3*a+k] * T_exp_1[243*e+81*l+27*f+9*h+3*d+i]) * Trace(P_exp_0[27*b+9*j+3*c+g]);
tmp[2] += sign * Trace(cgs * P_exp_1[27*a+9*h+3*d+i]) * Trace(T_exp_2[243*b+81*k+27*e+9*l+3*f+j] * S_tmp_T_CG[3*c+g]);
tmp[3] -= sign * Trace(cgs * P_exp_1[27*a+9*h+3*d+i]) * Trace(P_exp_0[27*b+9*j+3*c+g]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[4] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+j] * T_exp_3[243*c+81*g+27*b+9*h+3*d+i]);
tmp[5] += sign * Trace(cgs * T_exp_0[243*a+81*j+27*c+9*g+3*b+h] * S_tmp_T[3*d+i]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[6] -= sign * Trace(cgs * T_exp_2[243*a+81*j+27*f+9*g+3*e+l] * T_exp_4[243*b+81*k+27*c+9*h+3*d+i]);
tmp[7] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+g] * T_exp_4[243*b+81*j+27*c+9*h+3*d+i]);
tmp[8] += sign * Trace(cgs * P_exp_1[27*a+9*h+3*d+i]) * Trace(T_exp_2[243*b+81*k+27*e+9*l+3*f+g] * S_tmp_T_CG[3*c+j]);
tmp[9] += sign * Trace(cgs * P_exp_1[27*a+9*h+3*d+i]) * Trace(T_exp_2[243*b+81*j+27*f+9*g+3*e+l] * S_tmp_T_CG[3*c+k]);
tmp[10] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+g] * T_exp_3[243*c+81*j+27*b+9*h+3*d+i]);
tmp[11] -= sign * Trace(cgs * T_exp_2[243*a+81*j+27*f+9*g+3*e+l] * T_exp_3[243*c+81*k+27*b+9*h+3*d+i]);
tmp[12] -= sign * Trace(cgs * T_exp_2[243*a+81*g+27*f+9*j+3*e+l] * T_exp_4[243*b+81*k+27*c+9*h+3*d+i]);
tmp[13] += sign * Trace(cgs * S_tmxp_CG[3*a+g] * T_exp_4[243*b+81*j+27*c+9*h+3*d+i]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[15] -= sign * Trace(cgs * T_exp_5[243*a+81*g+27*b+9*j+3*c+k] * T_exp_1[243*e+81*l+27*f+9*h+3*d+i]);
tmp[16] += sign * Trace(cgs * T_exp_0[243*a+81*g+27*c+9*j+3*b+h] * S_tmp_T[3*d+i]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[19] += sign * Trace(cgs * S_tmxp_CG[3*a+j] * T_exp_4[243*b+81*g+27*c+9*h+3*d+i]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[22] -= sign * Trace(cgs * P_exp_1[27*a+9*h+3*d+i]) * Trace(P_exp_0[27*b+9*g+3*c+j]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[24] -= sign * Trace(cgs * T_exp_2[243*a+81*j+27*c+9*g+3*b+i] * T_exp_6[243*f+81*h+27*e+9*l+3*d+k]);
tmp[25] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+h] * T_exp_6[243*b+81*i+27*c+9*g+3*d+j]);
tmp[26] -= sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+g] * T_exp_6[243*f+81*j+27*e+9*l+3*d+k]);
tmp[27] += sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+g] * S_tmxp_T[3*d+j]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[28] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+j] * T_exp_6[243*c+81*g+27*b+9*i+3*d+h]);
tmp[29] += sign * Trace(cgs * T_exp_2[243*a+81*j+27*c+9*g+3*b+i] * S_tmxp_T[3*d+h]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[30] += sign * Trace(cgs * T_exp_2[243*a+81*j+27*f+9*g+3*e+l] * S_tmxp_T[3*d+k]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[31] += sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+g] * S_tmxp_T[3*d+j]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[32] -= sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+j] * T_exp_6[243*f+81*g+27*e+9*l+3*d+k]);
tmp[33] -= sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+k] * T_exp_6[243*e+81*l+27*f+9*g+3*d+j]);
tmp[34] -= sign * Trace(cgs * T_exp_2[243*a+81*k+27*e+9*l+3*f+g] * T_exp_6[243*c+81*j+27*b+9*i+3*d+h]);
tmp[35] -= sign * Trace(cgs * T_exp_2[243*a+81*j+27*f+9*g+3*e+l] * T_exp_6[243*c+81*k+27*b+9*i+3*d+h]);
tmp[37] -= sign * Trace(cgs * P_exp_3[27*a+9*g+3*d+j]) * Trace(P_exp_2[27*b+9*i+3*c+h]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[39] += sign * Trace(cgs * P_exp_3[27*a+9*g+3*d+j]) * Trace(T_exp_7[243*b+81*i+27*f+9*h+3*e+l] * S_tmp_T_CG[3*c+k]);
tmp[43] -= sign * Trace(cgs * P_exp_3[27*a+9*j+3*d+g]) * Trace(P_exp_2[27*b+9*i+3*c+h]) * Trace(P_exp_2[27*e+9*l+3*f+k]);
tmp[48] -= sign * Trace(cgs * T_exp_2[243*a+81*j+27*c+9*g+3*b+i] * T_exp_3[243*f+81*h+27*e+9*k+3*d+l]);
tmp[49] += sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(T_exp_2[243*e+81*j+27*c+9*g+3*b+i] * S_tmp_T_CG[3*f+h]);
tmp[50] -= sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+g] * T_exp_3[243*f+81*j+27*e+9*k+3*d+l]);
tmp[52] += sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(T_exp_2[243*e+81*h+27*b+9*i+3*c+g] * S_tmp_T_CG[3*f+j]);
tmp[54] += sign * Trace(cgs * T_exp_0[243*a+81*j+27*f+9*g+3*e+k] * S_tmp_T[3*d+l]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[55] -= sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(P_exp_0[27*e+9*j+3*f+g]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[56] -= sign * Trace(cgs * T_exp_2[243*a+81*h+27*b+9*i+3*c+j] * T_exp_3[243*f+81*g+27*e+9*k+3*d+l]);
tmp[57] += sign * Trace(cgs * S_tmxp_CG[3*a+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]) * Trace(P_exp_0[27*e+9*j+3*f+g]);
tmp[58] += sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(T_exp_2[243*e+81*h+27*b+9*i+3*c+j] * S_tmp_T_CG[3*f+g]);
tmp[59] -= sign * Trace(cgs * T_exp_0[243*a+81*j+27*f+9*g+3*e+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]);
tmp[60] += sign * Trace(cgs * T_exp_0[243*a+81*g+27*f+9*j+3*e+k] * S_tmp_T[3*d+l]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[62] -= sign * Trace(cgs * T_exp_2[243*a+81*g+27*c+9*j+3*b+i] * T_exp_3[243*f+81*h+27*e+9*k+3*d+l]);
tmp[63] -= sign * Trace(cgs * T_exp_5[243*a+81*g+27*e+9*j+3*f+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]);
tmp[65] -= sign * Trace(cgs * T_exp_0[243*a+81*g+27*f+9*j+3*e+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]);
tmp[66] -= sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(P_exp_0[27*e+9*g+3*f+j]) * Trace(P_exp_2[27*b+9*i+3*c+h]);
tmp[68] += sign * Trace(cgs * P_exp_1[27*a+9*k+3*d+l]) * Trace(T_exp_2[243*e+81*g+27*c+9*j+3*b+i] * S_tmp_T_CG[3*f+h]);
tmp[69] -= sign * Trace(cgs * T_exp_5[243*a+81*j+27*e+9*g+3*f+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]);
tmp[71] += sign * Trace(cgs * S_tmxp_CG[3*a+h] * T_exp_1[243*b+81*i+27*c+9*k+3*d+l]) * Trace(P_exp_0[27*e+9*g+3*f+j]);
tmp[96] += sign * Trace(P_exp_0[27*b+9*j+3*c+g]) * Trace(T_exp_2[243*e+81*k+27*d+9*l+3*a+i] * S_tmp_T_CG[3*f+h]);
tmp[97] -= sign * Trace(T_exp_2[243*b+81*k+27*d+9*l+3*a+i] * T_exp_8[243*f+81*h+27*e+9*j+3*c+g]);
tmp[98] -= sign * Trace(T_exp_2[243*b+81*h+27*a+9*i+3*d+l] * T_exp_9[243*e+81*k+27*f+9*j+3*c+g]);
tmp[100] -= sign * Trace(T_exp_2[243*b+81*k+27*d+9*l+3*a+i] * T_exp_9[243*e+81*h+27*f+9*j+3*c+g]);
tmp[104] -= sign * Trace(T_exp_2[243*b+81*h+27*a+9*i+3*d+l] * T_exp_9[243*e+81*k+27*f+9*g+3*c+j]);
tmp[105] += sign * Trace(T_exp_2[243*b+81*h+27*a+9*i+3*d+l] * S_tmp_T_CG[3*c+k]) * Trace(P_exp_0[27*e+9*j+3*f+g]);
tmp[106] -= sign * Trace(T_exp_2[243*b+81*k+27*d+9*l+3*a+i] * T_exp_9[243*e+81*h+27*f+9*g+3*c+j]);
tmp[107] -= sign * Trace(T_exp_0[243*b+81*j+27*f+9*g+3*e+h] * T_exp_10[243*a+81*i+27*d+9*l+3*c+k]);
tmp[111] -= sign * Trace(T_exp_5[243*b+81*g+27*e+9*j+3*f+h] * T_exp_10[243*a+81*i+27*d+9*l+3*c+k]);
tmp[117] -= sign * Trace(T_exp_5[243*b+81*j+27*e+9*g+3*f+h] * T_exp_10[243*a+81*i+27*d+9*l+3*c+k]);