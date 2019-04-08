function fig = itnnupdatefigure(net, i, fig, error, gradient, option)
%ITNNUPDATEFIGURE Plot errors, gradients at different phase.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)

if i == 1
    return;
end

interval = 1;
if 2^(fix(log2(i) + 1))  > 500
    interval = fix(log2(i));
end

figure(fig)
fig.Name = sprintf('input training neural network %s performance', option.phase);

errorSubPlot = subplot(2, 1, 1); % Plot errors.
ind = 1 : interval: i;
xplot = ind';
eplot = error(ind)';
plot(errorSubPlot, xplot, eplot, 'b--o');
ax = fig.CurrentAxes;
ax.Title.String = sprintf('%s errors = %3.6f, at %d epoch', option.phase, error(i), i);
ax.XLim = [0, i + 10];
ax.XLabel.String = 'Epochs';
ax.YLabel.String = sprintf('Erros (%s)',lower(net.performFcn));
legend(ax, option.phase, 'Location', 'NE');

gradientSubPlot = subplot(2, 1, 2); % Plot gradients.
gplot = gradient(ind)';
plot(gradientSubPlot, xplot, gplot, 'k:');
ax = fig.CurrentAxes;
ax.Title.String = sprintf('gradient = %3.6f, at %d epoch', gradient(i), i);
ax.XLim = [0, i + 10];
ax.XLabel.String = 'Epochs';
ax.YLabel.String = 'Gradient';
drawnow
end