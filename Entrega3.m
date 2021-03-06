E = imread('your-test-image.jpg');
A = imresize(E, 0.3);
figure();imshow(A);
Alab = rgb2lab(A);
figure();
imshow(uint8(Alab(:,:,2)));
B = im2bw(uint8(Alab(:,:,2)),0.09);
figure();
imshow(B);
C = imfill(B,'holes');
figure();
imshow(C);
D(:,:,1) = A(:,:,1) .* uint8(C);
D(:,:,2) = A(:,:,2) .* uint8(C);
D(:,:,3) = A(:,:,3) .* uint8(C);
figure();
imshow(D);

%HOUGH

[centro, radio] = imfindcircles(B,[50 500]);
x = round(centro(1,1));
y = round(centro(1,2));
r = round(radio);
[m,n] = size(B);
B3 = zeros(m,n);

for i = (y-r):(y+r)
  j = round(x + (sqrt((r*r)-((i-y)*(i-y)))));
  B3(i,j) = 1;
  
  j = round(x - (sqrt((r*r)-((i-y)*(i-y)))));
  B3(i,j) = 1;
  
end

for j = (x-r):(x+r)
  i = round(y + (sqrt((r*r)-((j-x)*(j-x)))));
  B3(i,j) = 1;
  
  i = round(y - (sqrt((r*r)-((j-x)*(j-x)))));
  B3(i,j) = 1;
end

figure(); imshow(B3);
B4 = imfill(B3,'holes');

BF = im2bw(B4) & C;
figure()
imshow(BF);

W(:,:,1) = A(:,:,1).* uint8(BF);
W(:,:,2) = A(:,:,2).* uint8(BF);
W(:,:,3) = A(:,:,3).* uint8(BF);

figure()
imshow(W);

% Etiquetado de clases
% Se inicia con una segmentacion complementaria, para separar los objetos
% de interes de la se�al de transito(numero o figuras), del resto de
% objetos

V(:,:,1) = W(y-radio:y+radio, x-radio:x+radio,1);
V(:,:,2) = W(y-radio:y+radio, x-radio:x+radio,2);
V(:,:,3) = W(y-radio:y+radio, x-radio:x+radio,3);

VT(:,:,1) = BF(y-radio:y+radio, x-radio:x+radio,1);

figure(); 
imshow(V);


T = graythresh(V);
V2 = not(im2bw(V, T));
figure();imshow(V2);

J = imdilate(V2,ones(3,3));
h = ones(3,3)/9; 
J = filter2(h, J);
figure(); imshow(not(J));

JT = imdilate(VT,ones(3,3));
h = ones(3,3)/9; 
JT = filter2(h, JT);
figure(); imshow(JT)

JF = J.*JT; 
[L, z] = bwlabel(JF,4);
figure();
imshow(L, [ ]);

%Boundingbox antes de la seleccion de clases
propied = regionprops(L, 'basic');
hold on
for q=1:size(propied,1)
    rectangle('Position',propied(q).BoundingBox, 'EdgeColor','g','LineWidth',1);
end

s = find([propied.Area]<8000);

hold on
for q=1:size(s,2)
    rectangle('Position',propied(s(q)).BoundingBox, 'EdgeColor','r','LineWidth',1);
end
BZ = zeros(size(L));

hold on
for q=1:size(s,2)
    f= round(propied(s(q)).BoundingBox);
    BZ(f(2):f(2)+f(4),f(1):f(1)+f(3))= J(f(2):f(2)+f(4),f(1):f(1)+f(3));
end
imshow(BZ);

[L2, z2] = bwlabel(BZ,4);
figure();
imshow(L2, [ ]);

%Extraccion de caracteristicas
[d,D] = Bio_labelregion(BZ,L2,3);

b(1).name = 'basicgeo'; b(1).options.show=1;
b(2).name = 'hugeo'; b(2).options.show=1;
b(3).name = 'flusser'; b(3).options.show=1;

op.b = b;
[X,Xn] = Bfx_geo(L2, op);
[X,Xn] = Bft_norm(X,Xn);

