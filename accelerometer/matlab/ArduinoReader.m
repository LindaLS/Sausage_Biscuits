% This class is used to read values dumped by the MPU6050 accelerometer
% on to the Arduino serial port.
classdef ArduinoReader < handle
    properties
        arduino_port;
        port;
        baud_rate;
        
        quaternian;
        acceleration;
        gravity;
        velocity;
        displacement;

        calibration;
    end

    methods
        function [obj] = ArduinoReader(port, baud_rate)
            obj.port = port;
            obj.baud_rate = baud_rate;
            obj.quaternian = [0 0 0 0]';
            obj.acceleration = [0 0 0]';
            obj.gravity = [0 0 9.81]';
            obj.velocity = [0 0 0]';
            obj.displacement = [0 0 0]';
            obj.calibration = [1 0 1 0 1 0]';
            
            % Open port used by Arduino
            obj.arduino_port = serial(obj.port, 'BaudRate', obj.baud_rate, 'BytesAvailableFcn', {@readCallback,obj});
            obj.arduino_port.BytesAvailableFcnCount = 22;
            obj.arduino_port.BytesAvailableFcnMode = 'byte';

            % Define callback function
            function readCallback(obj, event, arduino_reader)
                if (get(obj, 'BytesAvailable') ~= 0)
                    arduino_reader.quaternian = fread(obj, 4, 'float');
                    arduino_reader.acceleration = fread(obj, 3, 'int16');
                end
            end
            
            % Start reading from Arduino port
            fopen(obj.arduino_port);
            pause(2);
            fwrite(obj.arduino_port, 1);
        end

        function [] = close(obj)
            fclose(obj.arduino_port);
        end

        function [q, a] = raw_read(obj)
            q = obj.quaternian;
            a = obj.acceleration ./ 1000;
        end

        function [R, a_real] = read(obj)
            [q, a] = obj.raw_read();
            R = quaternion_to_r(q);
            g = R'*obj.gravity;
            a_real = zeros(3, 1);
            a_real(1) = a(1)*obj.calibration(1) + obj.calibration(2) - g(1);
            a_real(2) = a(2)*obj.calibration(3) + obj.calibration(4) - g(2);
            a_real(3) = a(3)*obj.calibration(5) + obj.calibration(6) - g(3);
        end

        % Calibration function
        % Calibrate accelerometer readings
        % TODO:Calculate sample rate
        function [] = calibrate(obj)
            N = 100;
            delta = 0.5;

            sample_num = 0;
            M = zeros(3*N, 6);
            y = zeros(3*N, 1);
            lamda = 0.1;
            while sample_num < N
                [q,a] = obj.raw_read;
                magnitude = sqrt(sum(a.^2));
                
                if (magnitude > (obj.gravity(3)-delta) && magnitude < (obj.gravity(3)+delta))
                    R = quaternion_to_r(q);
                    g = R'*obj.gravity;
                    M(3*sample_num+1, 1) = a(1);
                    M(3*sample_num+1, 2) = 1;
                    M(3*sample_num+2, 3) = a(2);
                    M(3*sample_num+2, 4) = 1;
                    M(3*sample_num+3, 5) = a(3);
                    M(3*sample_num+3, 6) = 1;
                    y(3*sample_num+1:3*sample_num+3,1) = g;
                    sample_num = sample_num + 1;
                end
                
                pause(0.1);
            end

            % Calibrate using least sum of squares with penalization
            obj.calibration = pinv(M' * M + lamda * eye(6))*M' * y;
        end
    end

end
