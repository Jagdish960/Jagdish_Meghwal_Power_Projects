P=99100;
Vin=290;
fs=10000;
Vout=750;
Ioutmax=P/Vout;
delIL=0.01*Ioutmax*(Vout/Vin);
delVout=0.01*Vout;
L=Vin*(Vout-Vin)/(delIL*fs*Vout)
C=(Ioutmax*(1-(Vin/Vout)))/(fs*delVout)
R=Vout/Ioutmax
d=(Vout-Vin)/Vout
