% This is sample code used to test whether data can be properly read from the arduino.
% It will plot the gravity-compensated acceleration with calibrated values.

% Initialization
clc;
clear;
close all;

% Add all function contained in 'converters'
addpath('./converters');

% Open Arduino port
ar = ArduinoReader('COM10', 115200);

% Wait for readings to stabilize, then start calibration
disp('Waiting for readings to stabilize...');
pause(3)
disp('Calibrating...');
%ar.calibrate();
disp('Done Calibrating.');

% Create figure to plot calibrated acceleration
figure1 = figure;
axes1 = axes('Parent',figure1);

% Plot acceleration
tic;
while toc < 15 % For 15 seconds
    [R, a] = ar.read();
    b = a
    g = sqrt(sum(b.^2))
    
    hold('off')
    quiver3(0,0,0,10,0,0,'LineWidth',2);
    hold('on')
    quiver3(0,0,0,0,10,0,'LineWidth',2);
    quiver3(0,0,0,0,0,10,'LineWidth',2);
    quiver3(0,0,0,a(1),a(2),a(3),...
    'MarkerSize',25,...
    'MarkerEdgeColor',[1 0 0],...
    'MarkerFaceColor',[1 0 0],...
    'Marker','.',...
    'LineWidth',3);
    xlim(axes1,[-12 12]);
    ylim(axes1,[-12 12]);
    zlim(axes1,[-12 12]);

    pause(0.01)
end

% Close port
ar.close();
