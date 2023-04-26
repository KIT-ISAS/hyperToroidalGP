function [varargout] = periodicKernel(varargin)
  % Compute values and derivatives of the periodic kernel
  % Parameters:
  %   param (2 x 1 column vector)
  %     hyperparameters for defining the periodic kernel including 
  %     lengthscale l (scalar) and period p (scalar)
  %   x (1 x nx vector)
  %     each column represents one of the inputs
  %   z (1 x nz vector)
  %     each column represents one of the inputs
  % Returns:
  %   Kbase (nx x nz matrix)
  %     values of the kernel function evaluated at each pair of input (x,z)
  %   dhyp (nx x 2*nx)
  %     derivatives of the kernel function w.r.t. each hyperparameter

  % compute number of hyperparameters for the kernel
  if nargin < 2, varargout{1} = 2; return 
  end

  % get hyperparameters
  param = varargin{1};
  l = param(1);
  p = param(2);

  x = varargin{2}; 
  z = varargin{3};  

  nx = size(x,2);

  xz = pi.*(abs(x' - z))./p;
  sin_delta = sin(xz);
  sin_delta2 = sin_delta.^2;
  Kbase = exp(-2.*sin_delta2/l^2);
  varargout{1} = Kbase;

  if nargout > 1 
    dhyp = zeros(nx,nx*2);
    dhyp(:,1:nx) = 4.*Kbase.*sin_delta2./l^3;
    dhyp(:,nx+1:end) = 4.*Kbase.*sin_delta.*cos(xz).*xz./(p*l^2); 
    varargout{2} = dhyp;
  end
end