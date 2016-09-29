% Initialization
clear;
clc;
close all;

% Parameters
sensorNum = 1;
listenTime = 30; % in seconds

% load the BioRadio API using a MATLAB's .NET interface
current_dir = cd;
[ deviceManager , success ] = load_API([current_dir '\BioRadioSDK.dll']);
if ~sucess
    errordlg('Could not load BioRadio API.')
    return
end

% Connect to the BioRadio objects
for i = 1:sensorNum
    % Ideally we will start a thread to do everything in this loop

    % Search for available sensors and select one
    [ deviceName{i} , macID{i} , ok ] = BioRadio_Find( deviceManager );
    if ~ok
        errordlg('Please select a BioRadio.')
        return
    end

    % Connect to BioRadio
    [ myDevice{i}, success ] = BioRadio_Connect ( deviceManager , macID{i} , deviceName{i} );
    if ~success
        errordlg('Could not connect to BioRadio.')
        return
    end

end

% Get data from each section
for i = 1:sensorNum
    BioRadioData{i} = BioRadio_Stream( myDevice{i} , listenTime , deviceName{i} );
end

% Disconnect from each BioRadio
for i = 1:sensorNum
    BioRadio_Disconnect( myDevice{i} )
end

% Test plot
plot(1:length(BioRadioData{1}), BioRadioData{1});