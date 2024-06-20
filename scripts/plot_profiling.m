function plot_profiling(filepath)

benchmark = readtable( ...
    filepath, ...
    'VariableNamingRule', 'preserve', ...
    'NumHeaderLines', 5, ...
    'VariableNamesLine', 4, ...
    'VariableUnitsLine', 5 ...
);
gpu = benchmark(benchmark.Type == "GPU activities",:);

x = gpu.Name;
y = gpu.("Time(%)");
for i = 1:length(x)
    parts = strsplit(x{i}, '(');
    x(i) = parts(1);
end

figure(1)
clf
bar(x, y)
set(gca,'TickLabelInterpreter','none')
grid on
ylim([0, 100])
ylabel('GPU time (%)')

end
