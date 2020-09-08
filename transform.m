A=textread('weight_Time_50_layer_2.txt');
A_=fliplr(A);
A_=flipud(A_);
padding1=zeros(28,28);
padding2=zeros(28,1);
padding3=zeros(1,57);
B1=[A padding2 padding1];
B2=[padding1 padding2 A_];
B=[B1;padding3;B2];
C1=fftshift1(B);
C2=abs(fft2(C1));%we use abs because here imaginary term is zero
C2=ifftshift1(C2);
%C=(C2-min(min(C2)))-(max(max(C2))-min(min(C2)));
%C=int8(255*C);
imshow(int8(C2))