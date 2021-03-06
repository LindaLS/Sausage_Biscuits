% Convert from Quaternion to Rotation Matrix
function [R] = quaternion_to_r (q)
    R = zeros(3,3);
    R(1,1) = 1 - 2*q(3)*q(3) - 2*q(4)*q(4);
    R(1,2) = 2*q(2)*q(3) - 2*q(1)*q(4);
    R(1,3) = 2*q(2)*q(4) + 2*q(1)*q(3);
    R(2,1) = 2*q(2)*q(3) + 2*q(1)*q(4);
    R(2,2) = 1 - 2*q(2)*q(2) - 2*q(4)*q(4);
    R(2,3) = 2*q(3)*q(4) - 2*q(1)*q(2);
    R(3,1) = 2*q(2)*q(4) - 2*q(1)*q(3);
    R(3,2) = 2*q(3)*q(4) + 2*q(1)*q(2);
    R(3,3) = 1 - 2*q(3)*q(3) - 2*q(2)*q(2);
end