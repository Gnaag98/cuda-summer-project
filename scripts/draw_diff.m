clear
clc

tolerance = 0.1;

density_atomic = load("../output/density_atomic.csv");
density_shared = load("../output/density_shared.csv");
density_diff = density_atomic - density_shared;
density_error = double(density_diff > tolerance) - double(density_diff < -tolerance);

figure(1)
clf
heatmap(0:size(density_atomic,2)-1, size(density_atomic,1)-1:-1:0, flipud(density_atomic))
grid off
title('shared')

figure(2)
clf
heatmap(0:size(density_shared,2)-1, size(density_shared,1)-1:-1:0, flipud(density_shared))
grid off
title('shared')

figure(3)
clf
heatmap(0:size(density_diff,2)-1, size(density_diff,1)-1:-1:0, flipud(density_diff))
grid off
title('diff')

figure(4)
clf
heatmap(0:size(density_error,2)-1, size(density_error,1)-1:-1:0, flipud(density_error))
grid off
title('error')
