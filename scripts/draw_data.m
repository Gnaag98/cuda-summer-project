clear
clc

global left
global right
global top
global bottom
global cell_width
global cell_height

left = -1000;
right = 1000;
top = 2000;
bottom = -2000;
cell_width = 200;
cell_height = 200;

density = load("../output/density.csv");
positions = load("../output/positions.csv");

fig_1 = figure(1);
clf
hold on

%draw_density(density);
draw_grid(density);
scatter(positions(:,1), positions(:,2), 150, '.b')
hold off
axis padded
xlabel('x')
ylabel('y')

figure(2)
clf
%heatmap(0:size(density,2)-1, size(density,1)-1:-1:0, flipud(density))
%grid off
hold on
draw_density(density);
draw_grid(density);
hold off
axis padded
xlabel('x')
ylabel('y')

function draw_density(density)
global left
global bottom
global cell_width
global cell_height

max_density = max(density, [], 'All');

scale = 1;

for row = 0:size(density,1)-1
    for column = 0:size(density,2)-1
        x = left + (column - 0.5 * scale) * cell_width;
        y = bottom + (row - 0.5 * scale) * cell_height;
        color = density(row+1, column+1) / max_density;
        rectangle('Position', [x, y, cell_width*scale, cell_height*scale], 'FaceColor',[1-color, 1-color, 1], 'LineStyle', 'none');
    end
end

end

function draw_grid(density)
global left
global right
global top
global bottom
global cell_width
global cell_height

for i = 0:size(density,1)-1
    y = bottom + i * cell_height;
    plot([left, right], [y, y], 'k')
end
plot([left, right], [top, top], 'k')

for i = 0:size(density,2)-1
    x = left + i * cell_width;
    plot([x, x], [bottom, top], 'k')
end
plot([right, right], [bottom, top], 'k')
end
