% Multichannel plotting.
% Currently only works with 2 outputs (0 and 1)
function [] = plot_emg(data)
    len = 0;
    ch = 0;

    % Get number of channels and length of data
    for i=1:length(data)
        temp = data(1,i);
        [ch, chunk] = size(temp{1});
        len = len + chunk;
    end

    data_array = zeros(ch, len);
    out_array = zeros(1, len);
    current_index = 1;
    for i=1:length(data)
        temp = data(1,i);
        temp = temp{1};
        [ch, chunk] = size(temp);
        out = data(2,i);

        data_array(:,current_index : current_index + chunk - 1) = temp;
        out_array(current_index : current_index + chunk - 1) = out{1};

        current_index = current_index + chunk;
    end

    figure;
    x = (1:len);

    current = out_array(1);
    indices = [];
    for i = 1:len    
        if(out_array(i) ~= current)
            current = out_array(i);
            indices = [indices i];
        end
    end
    indices = [indices len];

    for i = 1:ch
        subplot(ch,1,i);
        hold all;
        current = 1;
        for j = 1:length(indices)
            if(out_array(current) == 0)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'g');
            elseif(out_array(current) == 1)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'r');
            end
            current = indices(j);
        end
    end

end