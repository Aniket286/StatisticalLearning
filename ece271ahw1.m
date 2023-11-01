data = TrainsampleDCT_FG;

index_fg = zeros(size(data, 1), 1);
second_highest_idx = idx(:,2);
histogram(second_highest_idx)
hold on;

data = TrainsampleDCT_BG;

index_fg = zeros(size(data, 1), 1);
second_highest_idx = idx(:,2);
histogram(second_highest_idx)

% Add a legend and axis labels
legend('Array A', 'Array B');
xlabel('Value');
ylabel('Frequency');
title('Histogram of Two Arrays');