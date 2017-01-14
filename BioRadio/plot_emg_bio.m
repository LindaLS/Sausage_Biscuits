% Plot EMG data from BioRadio data files
% Input must be raw data imported from the output text files
function [] = plot_emg_bio(data)
    data = data';
    data_array = data(1,:);
    out_array = data(2,:);



    [ch, len] = size(data_array);

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
        ylim([-0.03 0.03])
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