% Multichannel plotting.
% Currently only works with 7 outputs (0 to 6)
function [] = plot_emg(data)
    [data_array, out_array] = read_emg_data(data)

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
            elseif(out_array(current) == 2)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'b');
            elseif(out_array(current) == 3)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'y');
            elseif(out_array(current) == 4)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'm');
            elseif(out_array(current) == 5)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'c');
            elseif(out_array(current) == 6)
                plot(x(current:indices(j)-1), data_array(i,current:indices(j)-1), 'k');
            end
            current = indices(j);
        end
    end

end