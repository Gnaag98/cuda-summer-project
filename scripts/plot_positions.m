clear
%clc

positions = load("../output/positions_atomic.csv");

U = 512;
V = 1024;

x = (positions(1,:) / 2048 + 0.5) * U;
y = (positions(2,:) / 4096 + 0.5) * V;

figure(5)
clf
plot(x, y, '*')
axis padded
axis equal
grid on
hold on
for i = 0 : V
    plot([0, U], [i, i], 'k')
end

for j = 0 : U
    plot([j, j], [0, V], 'k')
end
