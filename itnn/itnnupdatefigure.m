function figID = itnnupdatefigure(figID, i, data, option)
%ITNNUPDATEFIGURE Plot errors, gradients at different phase.

%   Date: Oct 06, 2018
%   Author: Zhongsheng Chen (E-mail:Zhongsheng.chen@outlook.com)

if i > 1
    x = 1 : i;
    % create legend for curves.
    legendStr = option.phase;
    % create plot data.
    xplot = x';
    yplot = data';
    % Plot data (erors or gradient).
    figure(figID);
    ax = gca;
    line = plot(ax, xplot, yplot);
    % Tiltle, label and legend
    figID.Name = sprintf('input training neural network %s performance', legendStr);
    line.Color = 'b'; line.LineStyle = '-'; line.Marker = 'o';      % Training error line
    ax.XLim = [0, i + 1];
    ax.XLabel.String = 'Epochs';
    ax.YLabel.String = option.plot.lineName;
    legend(ax, legendStr, 'Location', 'NE');
    drawnow
end