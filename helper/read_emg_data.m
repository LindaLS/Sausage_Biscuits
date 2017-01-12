% Multichannel plotting.
% Currently only works with 7 outputs (0 to 6)
function [data_array, out_array] = read_emg_data(data)
    % Get number of channels and length of data
    temp = data(1,1);
    [temp, ch] = size(temp{1});
    len = length(data);

    data_array = zeros(ch, len);
    out_array = zeros(1, len);
    for i=1:length(data)
        temp = data(1,i);
        temp = temp{1};
        out = data(2,i);

        data_array(:,i) = temp;
        out_array(i) = out{1};
    end
end